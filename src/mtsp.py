from pulp import *
import numpy as np
import networkx as nx
from itertools import combinations
import tsplib95 as tsplib
from tqdm import tqdm
import argparse

def solve_mtsp(start_positions, end_positions, weights):
    start_positions = np.array(start_positions)
    end_positions = np.array(end_positions)
    assert(start_positions.ndim == end_positions.ndim == 1)
    assert(start_positions.size == end_positions.size)
    A = start_positions.size
    
    weights = np.array(weights)    
    assert(weights.ndim == 2)
    assert(weights.shape[0] == weights.shape[1])
    N = weights.shape[0]
    
    assert(2 * A <= N)
    
    nodes = np.arange(N)
    agents = np.arange(A)

    print('creating model...')

    model = LpProblem('tsp', LpMinimize)
    variable_names = ['{{{}:{},{}}}'.format(a, u, v) for a in agents for u in nodes for v in nodes]
    variables = np.array(LpVariable.matrix('X', variable_names, cat=LpBinary)).reshape((A, N, N))

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

    obj_func = lpSum(variables * weights)
    model += obj_func

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
        
        
    print('creating subtour elimination inequalities...')
    for s in tqdm(range(2, N)):
        for a in agents:
            sa, ea = start_positions[a], end_positions[a]
            for S in combinations(nodes[np.logical_and(nodes != sa, nodes != ea)], s):
                S = list(S)
                subTourEl = lpSum(variables[(a,) + np.ix_(S, S)]) <= s - 1
                model += subTourEl, 'subtour elimination {} {}'.format(a, S)

    #print(model)

    model.solve()

    print([v.name for v in variables.reshape((-1,)) if v.value() == 1])
    
    paths = []
    for a in agents:
        path = []
        i = start_positions[a]
        while i != end_positions[a]:
            path.append(i)
            i = np.argmax([v.value() for v in variables[a, i]])
        path.append(i)
        paths.append(path)
        
    return paths
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary start and end points')
    parser.add_argument('--agents', default=3, type=int, help='number of agents')
    parser.add_argument('--nodes', default=10, type=int, help='number of nodes')

    args = parser.parse_args()
    N = args.nodes
    A = args.agents

    np.random.seed(42)

    G = nx.complete_graph(N).to_directed()
    weights = np.random.randint(1, 10, size=(N, N))
    weights_dict = {(u, v): {'weight': weights[u, v]} for u, v in G.edges}
    nx.set_edge_attributes(G, weights_dict)
    print('created random graph K', N)

    positions = np.random.choice(N, replace=False, size=2*A)
    start_positions = positions[:A]
    end_positions = positions[A:]
    print('start positions:', start_positions)
    print('end positions:', end_positions)
    
    paths = solve_mtsp(start_positions, end_positions, weights);
    
    for i, path in enumerate(paths):
        print('{}: {}'.format(i, path))
