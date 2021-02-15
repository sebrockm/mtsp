from pulp import *
import numpy as np
import networkx as nx
from itertools import combinations
import sys
import tsplib95 as tsplib
from tqdm import tqdm


np.random.seed(42)

G = None

if len(sys.argv) > 1:
    try:
        G = tsplib.load(sys.argv[1]).get_graph(normalize=True)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        weights = np.array([G.get_edge_data(*e)['weight'] for e in G.edges])
        print('loaded graph from file', sys.argv[1])
        print('Nodes:', N, 'Edges:', M)
    except:
        try:
            N = int(sys.argv[1])
        except:
            quit('Invalid Parameter')
else:
    N = 15
    
if not G:
    G = nx.complete_graph(N).to_directed()
    M = G.number_of_edges()
    weights = np.random.randint(1, 10, M)
    weights_dict = {e: {'weight': weights[i]} for i, e in enumerate(G.edges)}
    nx.set_edge_attributes(G, weights_dict)
    print('created random graph K', N)
    

print('creating model...')

model = LpProblem('tsp', LpMinimize)
variable_names = ['{' + str(u) + ',' + str(v) + '}' for u, v in G.edges]
variables = np.array(LpVariable.matrix('X', variable_names, cat=LpBinary))
variables_dict = {e: v for e, v in zip(G.edges, variables)}

obj_func = lpSum(variables * weights)
model += obj_func

print('creating degree inequalities...')

for n in G.nodes:
    inDegreeIneq = lpSum(variables_dict[(m, n)] for m in G.nodes if n != m) == 1
    outDegreeIneq = lpSum(variables_dict[(n, m)] for m in G.nodes if n != m) == 1
    model += inDegreeIneq, 'indegree inequality ' + str(n)
    model += outDegreeIneq, 'outdegree inequality ' + str(n)
    
print('creating subtour elimination inequalities...')

for r in tqdm(range(2, N)):
    for S in combinations(range(1, N), r):
        S = list(S)
        subTourEl = lpSum(variables_dict[(n,m)] for n in S for m in S if m != n) <= r - 1
        model += subTourEl, 'subtour elimination ' + str(S)

#print(model)

model.solve()

print([v.name for v in variables if v.value() == 1])