from pulp import EPS, LpStatusOptimal

def get_first_fractional_var_idx(variables):
    for i, v in enumerate(variables):
        if EPS <= v.value() <= 1 - EPS:
            return i
    return None

def branch_and_cut(lp, upper_bound = float('inf'), 
                   find_fractional_var_idx=get_first_fractional_var_idx
                   find_violated_constraints=None):
    S = [lp]
    lower_bound = EPS
    best_variables = None
    info_string = 'len(S): {:>4d}, lb: {:.5E} ub: {:.5E} GAP: {:>6.2%}'
    while len(S) > 0:
        print(info_string.format(len(S), lower_bound, upper_bound, upper_bound/lower_bound-1)
        current_lp = S.pop()
        current_lp.solve()
        if current_lp.status != LpStatusOptimal:
            continue
        
        current_lower_bound = current_lp.objective.value()
        lower_bound = max(lower_bound, current_lower_bound)

        if current_lower_bound >= upper_bound:
            continue

        variables = current_lp.variables()

        B = find_violated_constraints(variables)
        if len(B) > 0:
            for b in B:
                current_lp += b
            S.append(current_lp)
            continue

        i = find_fractional_var_idx(variables)
        if i is None:
            upper_bound = current_lower_bound
            best_variables = variables
            continue
            
        copy = current_lp.deep_copy()
        variables[i].setInitialValue(0)
        variables[i].fixValue()
        
        cVariables = copy.variables()
        cVariables[i].setInitialValue(1)
        cVariables[i].fixValue()
        S.extend([current_lp, copy])
        
    print(info_string.format(len(S), lower_bound, upper_bound, upper_bound/lower_bound-1)
    return best_variables, upper_bound