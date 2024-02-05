import numpy, networkx, tqdm
from queue import PriorityQueue
import matplotlib.pyplot as plot
from statistics import mode

# class
class Node:
    def __init__(self, i, state):
        self.i = i
        self.state = state
        self.edges = set()
    def link(self, node):
        if node not in self.edges:
            self.edges.add(node)
            node.link(self)
    # majority model algorithm
    def majority(self):
      if len(self.edges) != 0:
        self.state = mode([ node.state for node in self.edges ])
      else:
        self.state = self.state

colors = { 0: 'blue', 1: 'red'}

def generation(N,P): # number of nodes, probability
    global colors
    # the network graph
    graph = networkx.Graph()
    # the nodes dictionary and the edges list
    nodes, edges = { i: Node(i, numpy.random.choice([0, 1], size = None, p = [0.75, 0.25])) for i in numpy.arange(0, N) }, []
    # adding the nodes to the graph
    graph.add_nodes_from([key for key, node in nodes.items()])
    # generating edges between the nodes
    for i in tqdm.tqdm(range (N)):
        u = numpy.random.choice([key for key in nodes.keys()], size = None)
        w = numpy.random.choice([key for key in nodes.keys()], size = None)
        if u != w:
            edges.append((u,w))
            nodes[u].link(nodes[w])
        if numpy.random.binomial(n = 1, p = P, size = None):
            v = numpy.random.choice([key for key in nodes.keys()], size = None)
            if w != v:
                edges.append((v,w))
                nodes[v].link(nodes[w])
    # adding the edges to the graph
    graph.add_edges_from(edges)
    # color map according to the state variables
    nodecolors = [ colors[node.state] for node in nodes.values() ]
    # drawing the graph
    networkx.draw(graph, nodelist = nodes.keys(), node_size = 0.75, width = 0.5, node_color = nodecolors, alpha = 0.55)
    print(f'positive: {sum(1 for node in nodes.values() if node.state != 0)}, negative: {sum(1 for node in nodes.values() if node.state != 1)}')
    plot.show()
    return graph, nodes

def model(time, FES, lamda, nodes):
    IA = 0
    while IA == 0:
        IA = numpy.random.poisson(lam = lamda, size = None)
    FES.put((time + IA, "wakes up"))
    for key, node in nodes.items():
        node.majority()

def simulation (N, P, T, lamda):
    global colors
    (graph, nodes) = generation(N, P)
# the simulation time
    time = 0
# the Future Event Set
    FES = PriorityQueue()
# schedule the first arrival at t = 0
    FES.put((0, "wakes up"))
# Event Loop
    while time < T: # termination criteria: maximum simulation time (1440 minutes in a working day)
        (time, event) = FES.get()
        model(time, FES, lamda, nodes)
    return graph, nodes

for iter in [10, 100, 1000, 10000]:
    print(f'iter. {iter}')
    # number of nodes
    N = 1000
    # probability
    P = 0.0025
    (graph, nodes) = simulation(N, P, iter, 1)
    nodecolors = [ colors[node.state] for node in nodes.values() ]
    networkx.draw(graph, nodelist = nodes.keys(), node_size = 0.75, width = 0.5, node_color = nodecolors, alpha = 0.5)
    print(f'positive: {sum(1 for node in nodes.values() if node.state != 0)}, negative: {sum(1 for node in nodes.values() if node.state != 1)}')
    plot.show()