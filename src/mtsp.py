from pulp import *
import numpy as np
import networkx as nx
from itertools import product
import tsplib95 as tsplib
from tqdm import tqdm
import argparse
from bnc import branch_and_cut

optimization_modes = ['sum', 'max']

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

    print('creating model...')

    model = LpProblem('tsp', LpMinimize)
    variable_names = ['{{{},{},{}}}'.format(a, u, v) for a in agents for u in nodes for v in nodes]
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
        
        
    def find_violated_constraints(X):
        variables = np.array([X['X_{{{},{},{}}}'.format(a, u, v)] for a, u, v in product(agents, nodes, nodes)]).reshape((A, N, N))
        is_fractional = False
        Gall = nx.DiGraph()
        helpGraphs = []
        for a in agents:
            Gall.add_edge('dummy_source', start_positions[a], capacity=float('inf'))
            Gall.add_edge(end_positions[a], 'dummy_target', capacity=float('inf'))
            G = nx.DiGraph()
            for u, v in product(nodes, nodes):
                weight = variables[a, u, v].value()
                if weight > EPS:
                    if weight < 1 - EPS:
                        is_fractional = True
                    G.add_edge(u, v)
                    if Gall.has_edge(u, v):
                        Gall.edges[u, v]['capacity'] += weight
                    else:
                        Gall.add_edge(u, v, capacity=weight)
            helpGraphs.append(G)
        
        violated_constraints = []
        if is_fractional:
            # each agent has to pass a unit from dummy_source to dummy_target 
            min_cut, (V, W) = nx.minimum_cut(Gall, 'dummy_source', 'dummy_target')
            if min_cut < A - EPS:
                print('violated min cut')
                assert not ('dummy_source' in V or 'dummy_target' in W)
                violated_constraints.append(lpSum(variables[np.ix_(agents, list(V), list(W))]) >= A)
        else:
            for G in helpGraphs:
                # feasable solutions cannot have strongly connected components
                for comp in nx.strongly_connected_components(G):
                    if len(comp) > 1:
                        comp = list(comp)
                        print('found strongly connected component')
                        violated_constraints.append(lpSum(variables[np.ix_(agents, comp, comp)]) <= len(comp) - 1)
        return violated_constraints

    print(model)

    result_vars, _ = branch_and_cut(model, find_violated_constraints=find_violated_constraints)
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
    parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary start and end points')
    parser.add_argument('--agents', default=3, type=int, help='number of agents')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes')
    parser.add_argument('--mode', default='sum', choices=optimization_modes)

    args = parser.parse_args()
    N = args.nodes
    A = args.agents
    mode = args.mode

    np.random.seed(42)

    G = nx.complete_graph(N).to_directed()
    weights = np.random.randint(1, 100, size=(N, N))
    weights_dict = {(u, v): {'weight': weights[u, v]} for u, v in G.edges}
    nx.set_edge_attributes(G, weights_dict)
    print('created random graph K', N)

    positions = np.random.choice(N, replace=False, size=2*A)
    start_positions = positions[:A]
    end_positions = positions[A:]
    print('start positions:', start_positions)
    print('end positions:', end_positions)
    
    paths, lengths = solve_mtsp(start_positions, end_positions, weights, mode);
    
    for i, (path, length) in enumerate(zip(paths, lengths)):
        print('{}: {} length={}'.format(i, path, length))
