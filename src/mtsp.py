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

        return ret, time2 - time1
    return wrap

optimization_modes = ['sum', 'max']

@timing
def solve_mtsp(start_positions, end_positions, weights, optimization_mode='sum', time_limit=float('inf'), verbosity=1):
    start_time = time.time()
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

    if verbosity >= 2:
        print('creating model...')

    model = LpProblem('tsp', LpMinimize)
    variable_names = ['{{{},{},{}}}'.format(a, u, v) for a, u, v in product(agents, nodes, nodes)]
    variables = np.array(LpVariable.matrix('X', variable_names, 0, 1)).reshape((A, N, N))

    if verbosity >= 2:
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

    if verbosity >= 2:
        print('creating degree inequalities...')
    for n in nodes:
        model += lpSum(variables[:, :, n]) == 1
        model += lpSum(variables[:, n, :]) == 1
        if n not in start_positions:
            for a in agents:
                model += lpSum(variables[a, :, n]) == lpSum(variables[a, n, :])
    
    if verbosity >= 2:
        print('special inequalities for start and end nodes')
    for a in agents:
        model += lpSum(variables[a, start_positions[a], :]) == 1 # arcs out of start nodes
        model += lpSum(variables[a, :, end_positions[a]]) == 1 # arcs into end nodes
        model += variables[a, end_positions[a], start_positions[(a + 1) % A]] == 1 # artificial connections from end to next start
    
    if verbosity >= 2:
        print('inequalities to disallow cycles of length 2')
    for u in nodes:
        for v in range(u + 1, N):
            model += lpSum(variables[:, u, v] + variables[:, v, u]) <= 1
    
    def find_violated_constraints(X):
        variables = np.array([X['X_{{{},{},{}}}'.format(a, u, v)] for a, u, v in product(agents, nodes, nodes)]).reshape((A, N, N))
        
        # create undirected support graph
        G = nx.Graph()
        for u in nodes:
            for v in range(u+1, N):
                weight = lpSum(variables[:, [u, v], [v, u]]).value()
                assert -EPS < weight < 1 + EPS
                weight = max(0, min(1, weight))
                G.add_edge(u, v, weight=weight, capacity=min(weight, 1 - weight))
        
        G_sup = nx.subgraph_view(G, filter_edge=lambda u, v: G.edges[u, v]['weight'] > EPS)
        
        violated_constraints = []
        
        components = list(nx.connected_components(G_sup))
        if len(components) == 1: # there is only one component, so identify fractional subtours using undirected cut
            cut_size, (U, V) = nx.stoer_wagner(G_sup, weight='weight')
            if cut_size < 2 - EPS:
                ucut = lpSum(variables[np.ix_(agents, U, V)]) + lpSum(variables[np.ix_(agents, V, U)]) >= 2
                violated_constraints.append(ucut)
                if verbosity >= 2:
                    print(f'found violated ucut with cut size {cut_size}')
        else: # there is more than one component, so require them to be connected to the rest via two edges
            for component in components:
                U = list(component)
                V = list(set(nodes) - component)
                ucut = lpSum(variables[np.ix_(agents, U, V)]) + lpSum(variables[np.ix_(agents, V, U)]) >= 2
                violated_constraints.append(ucut)
            if verbosity >= 2:
                print('found several connected components')
        
        for component in components:
            # identifying two-matching inequalities
            E_gr = nx.subgraph_view(G_sup, filter_edge=lambda u, v: G_sup.edges[u, v]['weight'] > 0.5).edges
            odd = np.array([len(E_gr(v)) % 2 for v in nodes])
            def is_odd(nodes):
                return np.sum(odd[list(nodes)]) % 2 == 1
            def capacity(edges):
                return np.sum([G_sup.edges[e]['capacity'] for e in edges])
            
            T = nx.gomory_hu_tree(nx.subgraph_view(G_sup, filter_node=nx.filters.show_nodes(component)), capacity='capacity')
            for e in T.edges:
                S_i, V_S_i = nx.connected_components(nx.subgraph_view(T, filter_edge=nx.filters.hide_edges([e])))
                
                cut_edges = set(nx.edge_boundary(G_sup, S_i))
                cut_size = capacity(cut_edges)
                assert cut_size >= 0
                F = {e for e in cut_edges if e in E_gr}
                
                if is_odd(S_i) and cut_size < 1 - EPS:
                    assert len(F) % 2 == 1
                    all_cut_edges_but_F = [(u, v) for u, v in nx.edge_boundary(G, S_i) if (u, v) not in F and (v, u) not in F]
                    lhs = lpSum(variables[:, u, v] + variables[:, v, u] for u, v in all_cut_edges_but_F)
                    rhs = lpSum(variables[:, u, v] + variables[:, v, u] for u, v in F) - len(F) + 1
                    violated_constraints.append(lhs >= rhs)
                    if verbosity >= 2:
                        print('found violated comb inequality')
                    
                E1 = F
                E2 = {e for e in cut_edges if e not in E_gr}
                
                w1, e1 = min((G_sup.edges[e]['weight'], e) for e in E1) if len(E1) > 0 else (1, None)
                w2, e2 = max((G_sup.edges[e]['weight'], e) for e in E2) if len(E2) > 0 else (0, None)
                
                if not is_odd(S_i) and cut_size + min(w1 - (1 - w1), (1 - w2) - w2) < 1 - EPS:
                    if w1 - (1 - w1) < (1 - w2) - w2:
                        F -= {e1}
                    else:
                        F |= {e2}
                    assert len(F) % 2 == 1
                    all_cut_edges_but_F = [(u, v) for u, v in nx.edge_boundary(G, S_i) if (u, v) not in F and (v, u) not in F]
                    lhs = lpSum(variables[:, u, v] + variables[:, v, u] for u, v in all_cut_edges_but_F)
                    rhs = lpSum(variables[:, u, v] + variables[:, v, u] for u, v in F) - len(F) + 1
                    violated_constraints.append(lhs >= rhs)
                    if verbosity >= 2:
                        print('found violated comb inequality')
            
        return violated_constraints

    heuristic_paths = mtsp_heuristic()
    heuristic_solution = objective(heuristic_paths)
    paths_into_variables(heuristic_paths, variables)

    if verbosity >= 2:
        print(model)
    
    bnc_time_limit = time_limit - (time.time() - start_time)
    result_vars, (lb, ub) = branch_and_cut(model, find_violated_constraints=find_violated_constraints,
                                           upper_bound=heuristic_solution, time_limit=bnc_time_limit)
    for v in variables.reshape((-1,)):
        v.varValue = result_vars[v.name].value()
    
    result_values = np.array([v.value() for v in variables.reshape((-1,))]).reshape(variables.shape)
    used_edges = [v.name for v in variables.reshape((-1,)) if abs(v.value() - 1) < EPS]
    if verbosity >= 2:
        print(used_edges)
    gap = ub / lb - 1
    if verbosity >= 1:
        print(f'result: {ub} GAP: {gap:.2%}')
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
    
    return paths, lengths, gap
    

if __name__ == '__main__':
    import argparse
    from generate_qa_matrices import positions, generate_cost_matrix

    parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary start and end points')
    parser.add_argument('--agents', default=3, type=int, help='number of agents')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes')
    parser.add_argument('--mode', default='sum', choices=optimization_modes, help='minimize the sum (default) or maximum of all routes')
    parser.add_argument('--bench-file', default=None, help='if provided, performs a benchmark an writes the result into that file')
    parser.add_argument('--time-limit', default=float('inf'), type=float, help='time limit in seconds; if passed, a heuristic solution is returned')
    parser.add_argument('--verbosity', default=1, type=int, help='verbosity of the output; default is 1')

    args = parser.parse_args()
    bench_file = args.bench_file
    time_limit = args.time_limit
    verbosity = args.verbosity

    np.random.seed(42)
    
    if bench_file is not None:
        bench_nodes = range(5, args.nodes + 1)
        bench_agents = range(1, args.agents + 1)
        bench_modes = ['sum', 'max']
        repititions = 3
    else:
        bench_nodes = [args.nodes]
        bench_agents = [args.agents]
        bench_modes = [args.mode]
        repititions = 1
    
    for N in bench_nodes:
        weights = generate_cost_matrix(positions, N, verbosity=verbosity)
        for A in bench_agents:
            if 2 * A > N:
                continue
            special_positions = np.random.choice(N, replace=False, size=2*A)
            start_positions = special_positions[:A]
            end_positions = special_positions[A:]
            if verbosity >= 1:
                print('start positions:', start_positions)
                print('end positions:', end_positions)
            
            for mode in bench_modes:
                times = []
                results = []
                gaps = []
                for _ in range(repititions):
                    
                    (paths, lengths, gap), seconds = solve_mtsp(start_positions, end_positions, weights, mode, time_limit)
                    result = sum(lengths) if mode == 'sum' else max(lengths)
                    
                    if verbosity >= 1:
                        for i, (path, length) in enumerate(zip(paths, lengths)):
                            print('{}: {} length={}'.format(i, path, length))
                        
                    times.append(seconds)
                    results.append(result)
                    gaps.append(gap)
                
                result_string = f'N={N:>3d} A={A:>3d} mode={mode} time={np.mean(times):>7.3f}s result={np.min(results):>10d} gap={np.min(gaps):>7.2%}\n'
                if verbosity >= 1:
                    print(result_string)
                if bench_file is not None:
                    with open(bench_file, 'a') as f:
                        f.write(result_string)
