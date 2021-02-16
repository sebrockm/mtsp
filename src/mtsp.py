from pulp import *
import numpy as np
import networkx as nx
from itertools import combinations
import tsplib95 as tsplib
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Solving an mTSP with arbitrary stat and end points')
parser.add_argument('--agents', default=3, type=int, help='number of agents')
parser.add_argument('--nodes', default=10, type=int, help='number of nodes')

args = parser.parse_args()
N = args.nodes
A = args.agents

np.random.seed(42)

G = nx.complete_graph(N).to_directed()
M = G.number_of_edges()
weights = np.random.randint(1, 10, size=(N, N))
weights_dict = {(u, v): {'weight': weights[u, v]} for u, v in G.edges}
nx.set_edge_attributes(G, weights_dict)
print('created random graph K', N)

positions = np.random.choice(N, replace=False, size=2*A)
start_positions = positions[:A]
end_positions = positions[A:]
print('start positions:', start_positions)
print('end positions:', end_positions)

print('creating model...')

model = LpProblem('tsp', LpMinimize)
variable_names = ['{{{}:{},{}}}'.format(a, u, v) for a in range(A) for u in range(N) for v in range(N)]
variables = np.array(LpVariable.matrix('X', variable_names, cat=LpBinary)).reshape((A, N, N))

print('fixing some unused variables to zero...')
# self referring arc
for v in variables[:, np.arange(N), np.arange(N)].reshape((-1,)):
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
for a, p in enumerate(start_positions):
    for v in variables[np.arange(A) != a, p, :].reshape((-1,)):
        v.setInitialValue(0)
        v.fixValue()
    
# arcs into end nodes of wrong agents
for a, p in enumerate(end_positions):
    for v in variables[np.arange(A) != a, :, p].reshape((-1,)):
        v.setInitialValue(0)
        v.fixValue()

obj_func = lpSum(variables * weights)
model += obj_func

print('creating degree inequalities of start positions...')
for a, p in enumerate(start_positions):
    outStartIneq = lpSum(variables[a, p, :]) == 1
    model += outStartIneq, 'outdegree inequality of start position of {}'.format(a)

print('creating degree inequalities of end positions...')
for a, p in enumerate(end_positions):
    inEndIneq = lpSum(variables[a, :, p]) == 1
    model += inEndIneq, 'indegree inequality of end position of {}'.format(a)

print('creating degree inequalities for ordinary nodes...')
for n in range(N):
    if not n in start_positions and not n in end_positions:
        inDegreeIneq = lpSum(variables[:, :, n]) == 1
        outDegreeIneq = lpSum(variables[:, n, :]) == 1
        model += inDegreeIneq, 'indegree inequality ' + str(n)
        model += outDegreeIneq, 'outdegree inequality ' + str(n)
        for a in range(A):
            perAgentDegreesIneq = lpSum(variables[a, :, n]) == lpSum(variables[a, n, :])
            model += perAgentDegreesIneq, 'per agent degree inequality {} {}'.format(a, n)
    
    
print('creating subtour elimination inequalities...')
for s in tqdm(range(2, N)):
    for S in combinations(range(1, N), s):
        S = list(S)
        for a in range(A):
            subTourEl = lpSum(variables[(a,) + np.ix_(S, S)]) <= s - 1
            model += subTourEl, 'subtour elimination {} {}'.format(a, S)

#print(model)

model.solve()

print([v.name for v in variables.reshape((-1,)) if v.value() == 1])