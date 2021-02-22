from pulp import EPS, LpStatusOptimal, PULP_CBC_CMD
import multiprocessing

CPUS = multiprocessing.cpu_count()

def get_first_fractional_var_name(variables):
    for n, v in variables.items():
        if EPS <= v.value() <= 1 - EPS:
            return n
    return None

def branch_and_cut(lp, upper_bound = float('inf'), 
                   find_fractional_var_name=get_first_fractional_var_name,
                   find_violated_constraints=None):
    S = [lp]
    lower_bound = EPS
    best_variables = None
    info_string = 'len(S): {:>4d}, BOUNDS: [{:.5E}, {:.5E}] GAP: {:>6.2%}'
    while len(S) > 0:
        print(info_string.format(len(S), lower_bound, upper_bound, upper_bound/lower_bound-1))
        current_lp = S.pop()
        current_lp.solve(PULP_CBC_CMD(msg=False, threads=CPUS))
        if current_lp.status != LpStatusOptimal:
            continue
        
        current_lower_bound = current_lp.objective.value()
        lower_bound = max(lower_bound, current_lower_bound)

        if current_lower_bound >= upper_bound:
            continue

        variables = current_lp.variablesDict()
        
        B = find_violated_constraints(variables)
        if len(B) > 0:
            for b in B:
                current_lp += b
            S.append(current_lp)
            continue

        fractional_var = find_fractional_var_name(variables)
        if fractional_var is None:
            upper_bound = current_lower_bound
            best_variables = variables
            continue
            
        print('branching on fractional variable {} == {}'.format(fractional_var, variables[fractional_var].value()))
        copy = current_lp.deepcopy()
        variables[fractional_var].varValue = 0
        variables[fractional_var].fixValue()
        
        cVariables = copy.variablesDict()
        cVariables[fractional_var].varValue = 1
        cVariables[fractional_var].fixValue()
        S += [current_lp, copy]
        
    print(info_string.format(len(S), lower_bound, upper_bound, upper_bound/lower_bound-1))
    return best_variables, upper_bound