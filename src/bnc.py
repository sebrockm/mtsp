from pulp import EPS, LpStatusOptimal, LpStatus, PULP_CBC_CMD
import multiprocessing
from copy import deepcopy

CPUS = multiprocessing.cpu_count()

def get_first_fractional_var_name(variables):
    for n, v in variables.items():
        if n.startswith('X_') and EPS <= v.value() <= 1 - EPS:
            return n
    return None

def branch_and_cut(lp, upper_bound = float('inf'), 
                   find_fractional_var_name=get_first_fractional_var_name,
                   find_violated_constraints=None):
    S = [lp]
    best_variables = None
    info_string = 'len(S): {:>4d}, BOUNDS: [{:.5E}, {:.5E}] GAP: {:>6.2%}'
    while len(S) > 0:
        current_lp = S.pop()
        current_lp.solve(PULP_CBC_CMD(msg=False, threads=CPUS))
        if current_lp.status != LpStatusOptimal:
            print('Status: {}'.format(LpStatus[current_lp.status]))
            print('no feasible solution found for current LP, skipping it')
            continue
        
        current_lower_bound = current_lp.objective.value()
        print(info_string.format(len(S), current_lower_bound, upper_bound, upper_bound/current_lower_bound-1))

        if current_lower_bound >= upper_bound:
            print('lower bound of current LP is bigger than global upper bound, skipping')
            continue

        variables = current_lp.variablesDict()
        
        B = find_violated_constraints(variables)
        if len(B) > 0:
            for b in B:
                current_lp += b
            S.append(current_lp)
            print('added {} violated constraints, solving LP again'.format(len(B)))
            continue

        fractional_var = find_fractional_var_name(variables)
        if fractional_var is None:
            upper_bound = current_lower_bound
            best_variables = variables
            print('found feasible integer solution, updating upper bound')
            continue
            
        print('branching on fractional variable {} == {}'.format(fractional_var, variables[fractional_var].value()))

        copy = deepcopy(current_lp)       
        variables[fractional_var].varValue = 0
        variables[fractional_var].fixValue()
        
        cVariables = copy.variablesDict()
        cVariables[fractional_var].varValue = 1
        cVariables[fractional_var].fixValue()
        S += [current_lp, copy]
        
    print(info_string.format(len(S), current_lower_bound, upper_bound, upper_bound/current_lower_bound-1))
    return best_variables, upper_bound