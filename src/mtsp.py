from pulp import *
import numpy as np
import networkx as nx
from itertools import product
import tsplib95 as tsplib
from tqdm import tqdm
from bnc import branch_and_cut
import time

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} took {:.3f} s'.format(f.__name__, time2 - time1))

        return ret
    return wrap

optimization_modes = ['sum', 'max']

@timing
def solve_mtsp(start_positions, end_positions, weights, optimization_mode='sum'):
    assert optimization_mode in optimization_modes
    
    start_positions = np.array(start_positions)
    end_positions = np.array(end_positions)
    assert start_positions.ndim == end_positions.ndim == 1
    assert start_positions.size == end_positions.size
    A = start_positions.size
    
    weights = np.array(weights)
    assert weights.ndim == 2
    assert weights.shape[0] == weights.shape[1]
    N = weights.shape[0]
    
    assert 2 * A <= N
    
    nodes = np.arange(N)
    agents = np.arange(A)
    
    def objective(paths):
        sums = [np.sum(weights[paths[a][:-1], paths[a][1:]]) for a in agents]
        return sum(sums) if optimization_mode == 'sum' else max(sums)
    
    def mtsp_heuristic():
        paths = [[start_positions[a], end_positions[a]] for a in agents]
        for n in nodes:
            if n not in start_positions and n not in end_positions:
                min_obj, min_a, min_p = float('inf'), -1, None
                for a in agents:
                    for i in range(1, len(paths[a])):
                        tmp_path = paths[a][:i] + [n] + paths[a][i:]
                        obj = objective(paths[:a] + [tmp_path] + paths[a+1:])
                        if obj < min_obj:
                            min_obj, min_a, min_p = obj, a, tmp_path
                paths[min_a] = min_p
        return paths
        
    def paths_into_variables(paths, variables):
        for v in variables.reshape((-1,)):
            v.varValue = 0
        for a in agents:
            for v in variables[a, paths[a][:-1], paths[a][1:]]:
                v.varValue = 1

    print('creating model...')

    model = LpProblem('tsp', LpMinimize)
    variable_names = ['{{{},{},{}}}'.format(a, u, v) for a, u, v in product(agents, nodes, nodes)]
    variables = np.array(LpVariable.matrix('X', variable_names, 0, 1)).reshape((A, N, N))

    print('fixing some unused variables to zero...')
    # self referring arc (entries on diagonal)
    for v in variables[:, nodes, nodes].reshape((-1,)):
        v.setInitialValue(0)
        v.fixValue()

    # arcs into start nodes
    for v in variables[:, :, start_positions].reshape((-1,)):
        v.setInitialValue(0)
        v.fixValue()

    # arcs out of end nodes
    for v in variables[:, end_positions, :].reshape((-1,)):
        v.setInitialValue(0)
        v.fixValue()
        
    # arcs out of start nodes of wrong agents
    for a in agents:
        for v in variables[agents != a, start_positions[a], :].reshape((-1,)):
            v.setInitialValue(0)
            v.fixValue()
        
    # arcs into end nodes of wrong agents
    for a in agents:
        for v in variables[agents != a, :, end_positions[a]].reshape((-1,)):
            v.setInitialValue(0)
            v.fixValue()

    if optimization_mode == 'sum':
        obj_func = lpSum(variables * weights)
        model += obj_func
    elif optimization_mode == 'max':
        max_route_length = LpVariable('max rout length')
        model += max_route_length # objective function: minimize max route length
        for a in agents:
            model += max_route_length >= lpSum(variables[a] * weights)

    print('creating degree inequalities of start and end positions...')
    for a in agents:
        outStartIneq = lpSum(variables[a, start_positions[a], :]) == 1
        inEndIneq = lpSum(variables[a, :, end_positions[a]]) == 1
        model += outStartIneq, 'outdegree inequality of start position of {}'.format(a)
        model += inEndIneq, 'indegree inequality of end position of {}'.format(a)

    print('creating degree inequalities for ordinary nodes...')
    for n in nodes:
        if not n in start_positions and not n in end_positions:
            inDegreeIneq = lpSum(variables[:, :, n]) == 1
            outDegreeIneq = lpSum(variables[:, n, :]) == 1
            model += inDegreeIneq, 'indegree inequality ' + str(n)
            model += outDegreeIneq, 'outdegree inequality ' + str(n)
            for a in agents:
                perAgentDegreesIneq = lpSum(variables[a, :, n]) == lpSum(variables[a, n, :])
                model += perAgentDegreesIneq, 'per agent degree inequality {} {}'.format(a, n)    
    
    for u, v in product(nodes, nodes):
        model += lpSum(variables[:, u, v]) <= 1
        
    def find_violated_constraints(X):
        variables = np.array([X['X_{{{},{},{}}}'.format(a, u, v)] for a, u, v in product(agents, nodes, nodes)]).reshape((A, N, N))
        
        all_edges = {(u, v) for a, u, v in product(agents, nodes, nodes) if variables[a, u, v].value() > EPS}
        Gall = nx.DiGraph(all_edges)
        
        # identifying subtours
        violated_constraints = []
        for cycle in nx.simple_cycles(Gall):
            subset_length = lpSum(variables[np.ix_(agents, cycle, cycle)])
            #print('found cycle: {} cycle length: {} sum variables: {}'.format(cycle, len(cycle), subset_length.value()))
            if subset_length.value() > len(cycle) - 1 + EPS:
                violated_constraints.append(subset_length <= len(cycle) - 1)
                
        return violated_constraints

    heuristic_paths = mtsp_heuristic()
    heuristic_solution = objective(heuristic_paths)
    paths_into_variables(heuristic_paths, variables)

    #print(model)

    result_vars, _ = branch_and_cut(model, find_violated_constraints=find_violated_constraints, upper_bound=heuristic_solution)
    for v in variables.reshape((-1,)):
        v.varValue = result_vars[v.name].value()

    print([v.name for v in variables.reshape((-1,)) if v.value() == 1])
    
    paths = []
    lengths = []
    for a in agents:
        path = []
        length = 0
        i = start_positions[a]
        while i != end_positions[a]:
            path.append(i)
            prev_i = i
            i = np.argmax([v.value() for v in variables[a, i]])
            length += weights[prev_i, i]
        path.append(i)
        paths.append(path)
        lengths.append(length)
        
    return paths, lengths
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary start and end points')
    parser.add_argument('--agents', default=3, type=int, help='number of agents')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes')
    parser.add_argument('--mode', default='sum', choices=optimization_modes)

    args = parser.parse_args()
    N = args.nodes
    A = args.agents
    mode = args.mode

    np.random.seed(42)

    weights = np.random.randint(1, 100, size=(N, N))
    positions = np.random.choice(N, replace=False, size=2*A)
    start_positions = positions[:A]
    end_positions = positions[A:]
    print('start positions:', start_positions)
    print('end positions:', end_positions)
    
    paths, lengths = solve_mtsp(start_positions, end_positions, weights, mode);
    
    for i, (path, length) in enumerate(zip(paths, lengths)):
        print('{}: {} length={}'.format(i, path, length))
