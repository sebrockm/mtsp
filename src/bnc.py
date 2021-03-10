from pulp import EPS, LpStatusOptimal, LpStatus, PULP_CBC_CMD, LpProblem
import multiprocessing
from copy import deepcopy
import time
from queue import PriorityQueue

class S_data:
    def __init__(self, lower_bound=float('-inf'), fixed_vars={}):
        self.lower_bound = lower_bound
        self.fixed_vars = fixed_vars
        
    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

CPUS = multiprocessing.cpu_count()

def get_first_fractional_var_name(variables):
    for n, v in variables.items():
        if n.startswith('X_') and EPS <= v.value() <= 1 - EPS:
            return n
    return None

def get_05_fractional_var_name(variables):
    closest, min_abs = None, 1
    for n, v in variables.items():
        if n.startswith('X_') and EPS <= v.value() <= 1 - EPS:
            if abs(v.value() - 0.5) < min_abs:
                min_abs = abs(v.value() - 0.5)
                closest = n
                if min_abs == 0:
                    break
    return closest

def branch_and_cut(lp, upper_bound = float('inf'),
                   find_fractional_var_name=get_05_fractional_var_name,
                   find_violated_constraints=None,
                   exploit_fractional_solution=None,
                   time_limit=float('inf'),
                   verbosity=1):
    start_time = time.time()
    info_string = 'len(S): {:>4d}, BOUNDS: [{:.5E}, {:.5E}] GAP: {:>6.2%}'
    last_char = '\r' if verbosity == 1 else '\n'
    
    S = PriorityQueue()
    lower_bound = float('-inf')
    fixed_vars = {}
    S.put(S_data(lower_bound, fixed_vars))

    variables = lp.variablesDict()
    best_variables = deepcopy(variables)

    assert find_fractional_var_name(best_variables) is None
    
    while len(S.queue) > 0 and time.time() - start_time < time_limit:
        for name in fixed_vars.keys():
            variables[name].unfixValue()
        
        s_data = S.get()
        lower_bound = s_data.lower_bound
        fixed_vars = s_data.fixed_vars
        
        for name, value in fixed_vars.items():
            variables[name].varValue = value
            variables[name].fixValue()
        
        lp.solve(PULP_CBC_CMD(msg=False, threads=CPUS))
        
        if lp.status != LpStatusOptimal:
            if verbosity >= 2:
                print('Status: {}'.format(LpStatus[lp.status]))
                print('no feasible solution found for current LP, skipping it')
            continue

        variables = lp.variablesDict()
        
        if exploit_fractional_solution is not None:
            exploit_value, exploit_variables = exploit_fractional_solution(variables)
            if exploit_value < upper_bound:
                upper_bound = exploit_value
                best_variables = exploit_variables
                if verbosity >= 2:
                    print('exploiting: found better solution, updating upper bound')
        
        current_lower_bound = lp.objective.value()
        lower_bound = min(upper_bound, current_lower_bound)
        if len(S.queue) > 0:
            lower_bound = min(lower_bound, S.queue[0].lower_bound)
        
        if lower_bound >= upper_bound:
            break
        
        if verbosity >= 1:
            print(info_string.format(len(S.queue), lower_bound, upper_bound, upper_bound/lower_bound-1), end=last_char)

        if current_lower_bound >= upper_bound:
            if verbosity >= 2:
                print('lower bound of current LP is bigger than global upper bound, skipping')
            continue
        
        B = find_violated_constraints(variables)
        if len(B) > 0:
            for b in B:
                lp += b
            S.put(S_data(current_lower_bound, fixed_vars))
            if verbosity >= 2:
                print('added {} violated constraints, solving LP again'.format(len(B)))
            continue

        fractional_var = find_fractional_var_name(variables)
        if fractional_var is None:
            upper_bound = current_lower_bound
            best_variables = deepcopy(variables)
            if verbosity >= 2:
                print('found feasible integer solution, updating upper bound')
            assert lower_bound <= upper_bound + EPS, f'{lower_bound} <= {upper_bound}'
            if lower_bound >= upper_bound:
                break
            else:
                continue
        
        if verbosity >= 2:
            print('branching on fractional variable {} == {}'.format(fractional_var, variables[fractional_var].value()))

        S.put(S_data(current_lower_bound, fixed_vars | {fractional_var: 0}))
        S.put(S_data(current_lower_bound, fixed_vars | {fractional_var: 1}))
        
    assert find_fractional_var_name(best_variables) is None
    if verbosity >= 1:
        print(info_string.format(len(S.queue), lower_bound, upper_bound, upper_bound/lower_bound-1))
    return best_variables, (lower_bound, upper_bound)