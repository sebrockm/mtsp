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

        return ret, time2 - time1
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
    
    # artificially connect end positions i to start positions i+1 with zero cost
    weights[end_positions, list(start_positions[1:]) + [start_positions[0]]] = 0
    
    def objective(paths):
        sums = [np.sum(weights[paths[a][:-1], paths[a][1:]]) for a in agents]
        return sum(sums) if optimization_mode == 'sum' else max(sums)
    
    def mtsp_heuristic():
        paths = [[start_positions[a], end_positions[a]] for a in agents]
        for n in set(nodes) - set(start_positions) - set(end_positions):
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
            variables[a, paths[a][-1], paths[(a+1)%A][0]].varValue = 1
                
    def validate_result(paths):
        reached_nodes = [n for path in paths for n in path]
        assert sorted(reached_nodes) == list(nodes)
        for a in agents:
            assert paths[a][0] == start_positions[a]
            assert paths[a][-1] == end_positions[a]
        G = nx.DiGraph()
        for a in agents:
            for u, v in zip(paths[a], paths[a][1:]):
                G.add_edge(u, v)
            G.add_edge(end_positions[a], start_positions[(a+1)%A])
        cycles = list(nx.simple_cycles(G))
        assert len(cycles) == 1
        assert len(cycles[0]) == N

    print('creating model...')

    model = LpProblem('tsp', LpMinimize)
    variable_names = ['{{{},{},{}}}'.format(a, u, v) for a, u, v in product(agents, nodes, nodes)]
    variables = np.array(LpVariable.matrix('X', variable_names, 0, 1)).reshape((A, N, N))

    print('fixing some unused variables to zero...')
    # self referring arc (entries on diagonal)
    for v in variables[:, nodes, nodes].reshape((-1,)):
        model += v == 0

    if optimization_mode == 'sum':
        obj_func = lpSum(variables * weights)
        model += obj_func
    elif optimization_mode == 'max':
        max_route_length = LpVariable('max route length')
        model += max_route_length # objective function: minimize max route length
        for a in agents:
            model += max_route_length >= lpSum(variables[a] * weights)

    print('creating degree inequalities...')
    for n in nodes:
        model += lpSum(variables[:, :, n]) == 1
        model += lpSum(variables[:, n, :]) == 1
        if n not in start_positions:
            for a in agents:
                model += lpSum(variables[a, :, n]) == lpSum(variables[a, n, :])
    
    print('special inequalities for start and end nodes')
    for a in agents:
        model += lpSum(variables[a, start_positions[a], :]) == 1 # arcs out of start nodes
        model += lpSum(variables[a, :, end_positions[a]]) == 1 # arcs into end nodes
        model += variables[a, end_positions[a], start_positions[(a + 1) % A]] == 1 # artificial connections from end to next start
        
    def find_violated_constraints(X):
        variables = np.array([X['X_{{{},{},{}}}'.format(a, u, v)] for a, u, v in product(agents, nodes, nodes)]).reshape((A, N, N))
        
        # create directed support graph
        G_sup = nx.DiGraph()
        for u, v in product(nodes, nodes):
            weight = lpSum(variables[:, u, v]).value()
            if weight > EPS:
                G_sup.add_edge(u, v, weight=weight)
        
        violated_constraints = []
        
        # identifying subtours
        for cycle in nx.simple_cycles(G_sup):
            if len(cycle) >= N:
                continue
            shifted = cycle[1:] + [cycle[0]]
            length = lpSum(variables[:, cycle, shifted])
            if length.value() > len(cycle) - 1 + EPS:
                violated_constraints.append(length <= len(cycle) - 1)
                print('found subtour of {} nodes with length {}'.format(len(cycle), length.value()))
                
        # create undirected support graph
        G_sup = nx.Graph()
        for u in nodes:
            for v in range(u+1, N):
                weight = lpSum(variables[:, [u, v], [v, u]]).value()
                if weight > EPS:
                    G_sup.add_edge(u, v, weight=weight)
        
        # identifying two-matching inequalities
        for handle in nx.cycle_basis(G_sup):
            handle_length = lpSum(variables[np.ix_(agents, handle, handle)])
            weights_cut_edges = [(lpSum(variables[:, [u, v], [v, u]]), (u, v)) for u, v in product(handle, set(nodes) - set(handle))]
            weights_cut_edges = [(w, e) for w, e in weights_cut_edges if w.value() > EPS]
            weights_cut_edges = sorted(weights_cut_edges, key=lambda we: we[0].value(), reverse=True)
            for k in range(1, len(weights_cut_edges) + 1, 2):
                weights_teeth = weights_cut_edges[:k]
                comb_length = handle_length + lpSum(w for w, e in weights_teeth)
                if comb_length.value() > len(handle) + (k - 1) // 2 + EPS:
                    violated_constraints.append(comb_length <= len(handle) + (k - 1) // 2)
                    print('found comb with weight {} while {} is allowed'.format(comb_length.value(), len(handle) + (k - 1) // 2))
            
        return violated_constraints

    heuristic_paths = mtsp_heuristic()
    heuristic_solution = objective(heuristic_paths)
    paths_into_variables(heuristic_paths, variables)

    #print(model)

    result_vars, _ = branch_and_cut(model, find_violated_constraints=find_violated_constraints, upper_bound=heuristic_solution)
    for v in variables.reshape((-1,)):
        v.varValue = result_vars[v.name].value()
    
    result_values = np.array([v.value() for v in variables.reshape((-1,))]).reshape(variables.shape)
    used_edges = [v.name for v in variables.reshape((-1,)) if abs(v.value() - 1) < EPS]
    print(used_edges)
    assert len(used_edges) == N
    
    paths = []
    lengths = []
    for a in agents:
        path = []
        length = 0
        i = start_positions[a]
        while i != end_positions[a]:
            path.append(i)
            prev_i = i
            where = np.where(np.abs(result_values[a, i, :] - 1) < EPS)
            assert len(where[0]) == 1
            i = where[0][0]
            length += weights[prev_i, i]
        path.append(i)
        paths.append(path)
        lengths.append(length)
    
    validate_result(paths)
    
    return paths, lengths
    

if __name__ == '__main__':
    import argparse
    from generate_qa_matrices import positions, generate_cost_matrix

    parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary start and end points')
    parser.add_argument('--agents', default=3, type=int, help='number of agents')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes')
    parser.add_argument('--mode', default='sum', choices=optimization_modes)
    parser.add_argument('--bench-file', default=None, help='if provided, performs a benchmark an writes the result into that file')

    args = parser.parse_args()
    bench_file = args.bench_file

    np.random.seed(42)
    
    if bench_file is not None:
        bench_nodes = range(5, args.nodes + 1)
        bench_agents = range(1, args.agents + 1)
        bench_modes = ['sum', 'max']
        repititions = 10
    else:
        bench_nodes = [args.nodes]
        bench_agents = [args.agents]
        bench_modes = [args.mode]
        repititions = 1
    
    for N in bench_nodes:
        weights = generate_cost_matrix(positions, N)
        for A in bench_agents:
            if 2 * A > N:
                continue
            special_positions = np.random.choice(N, replace=False, size=2*A)
            start_positions = special_positions[:A]
            end_positions = special_positions[A:]
            print('start positions:', start_positions)
            print('end positions:', end_positions)
            
            for mode in bench_modes:
                times = []
                results = []
                for _ in range(repititions):
                    
                    (paths, lengths), seconds = solve_mtsp(start_positions, end_positions, weights, mode)
                    result = sum(lengths) if mode == 'sum' else max(lengths)
                    
                    for i, (path, length) in enumerate(zip(paths, lengths)):
                        print('{}: {} length={}'.format(i, path, length))
                        
                    times.append(seconds)
                    results.append(result)
                    
                assert results.count(results[0]) == len(results)
                with open(bench_file, 'a') as f:
                    f.write(f'N={N:>3d} A={A:>3d} mode={mode} time={sum(times)/len(times):>7.3f}s result={results[0]:>10d}\n')
