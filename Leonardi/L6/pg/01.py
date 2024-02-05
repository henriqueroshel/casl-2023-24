# ==================================================
import math, numpy, queue, networkx, copy, itertools, scipy.stats, collections
import matplotlib.pyplot as plot
# ==================================================
# the node class stores the node identification code, the node state, the set of the node's neighbors,
# and the 2-dimensional and 3-dimensional coordinates of the node
class Node:
    def __init__(self, i, state):
        self.i = i
        self.state = state
        self.neighbors = set()
    def D2(self, coordinate):
        self.x, self.y = coordinate
    def D3(self, coordinate):
        self.x, self.y, self.z = coordinate
    def degree(self):
        return numpy.array(list(self.neighbors)).size
# ==================================================
# the giant component (GC) search function via deep-first search (DFS) algorithm
# iterates recursively over each node of the network, looking for the subnetwork with the greatest number of nodes
def GC(N):
    def DFS(stack, node, V):
        V[node] = True
        stack.append(node)
        if N[node].neighbors:
            for neighbor in N[node].neighbors:
                if V[neighbor] == False:
                    stack = DFS(stack, neighbor, V)
        return stack
    V, G = [False for node in N.keys()], []
    for node in N.keys():
        if V[node] == False:
            stack = []
            G.append(DFS(stack, node, V))
    G = {len(array): array for array in G}
    ON = G[max(key for key in G.keys())]
    OFF = [node for node in N.keys() if ON.count(node) == 0]
    return ON, OFF
# returns the dictionary of nodes belonging to the giant component (ON)
# and the dictionary of nodes not belonging to the giant component (OFF)
# ==================================================
# the voter model implementation function uniformly and randomly selects a neighbor of the node and copy the state accordingly
def voter(N, node):
    if N[node].neighbors:
        neighbors = numpy.array(list(N[node].neighbors))
        N[node].state = N[numpy.random.choice(neighbors, size = None)].state
        status = sum(state for state in [N[node].state for node in neighbors])
        if status == neighbors.size or status == - neighbors.size:
            return numpy.random.choice([node for node in N.keys()], size = None)
        else:
            return numpy.random.choice(neighbors, size = None)
    else:
        return numpy.random.choice([node for node in N.keys()], size = None)
# in case of uniformity of the states among the neighbors, returns a random node of the network
# otherwise, returns a random node among the neighbors
# ==================================================
# the node wake-up function takes the return node from the Voter Model implementation function,
# selects the value of the lambda parameter according to the degrees of freedom of the node,
# generates a wake-up time according to lambda and inserts the parameters into the FES
def wakeup(time, FES, N, node):
    node = voter(N, node)
    L, IA = N[node].degree() if N[node].neighbors else 1, 0
    while IA == 0:
        IA = numpy.random.exponential(scale = 1 / L, size = None)
    FES.put((time + IA, node))
# ==================================================
# the G(n,p) generation function generates the network N (representing the dictionary of network nodes) according to the G(n,p) model,
# with edge probability equal to p for each pair of nodes in the network
def G(n, p):
    print(f'\nsimulating G({n},{10 / n}) with state probability = {p}')
    N = { i: Node(i, numpy.random.choice([+ 1, - 1], size = None, p = [p, 1 - p])) for i in range(n) }
    E = numpy.random.choice(range(n), size = (numpy.random.binomial(n = math.comb(n, 2), p = 10 / n), 2))
    C = 0
    for (v, w) in E:
        if v != w:
            C += 1; N[v].neighbors.add(w); N[w].neighbors.add(v)
    print(f'number of edges - theoretical: {math.comb(n, 2) * 10 / n}, empirical: {C}')
    ON, OFF = GC(N)
    O = { + 1: sum(1 for node in OFF if N[node].state == + 1),
          - 1: sum(1 for node in OFF if N[node].state == - 1) }
    print(f'giant component: {numpy.array(ON).size}, off-network nodes: {O}')
    N = { i: node for i, node in N.items() if i in ON }
    return N, E
# returns the dictionary of nodes belonging only to the Giant Component
# ==================================================
# the 2D Regular Grid generation function generates the network N (representing the dictionary of network nodes) according to the 2D Regular Grid model,
# with edge probability as a function of the distance between nodes
def D2(n, p):
    def distance(v, w):
        return abs(v.x - w.x) + abs(v.y - w.y)
    print(f'\nsimulating 2D grid with state probability = {p}')
    N = { i: Node(i, numpy.random.choice([+ 1, - 1], size = None, p = [p, 1 - p])) for i in range(n) }
    E = None
    def arange(n, k):
        return numpy.arange(start = 0, stop = math.pow(n, 1 / k))
    X, Y = arange(n, 2), arange(n, 2)
    for node, coordinate in zip(N.values(), [c for c in itertools.product(* [X, Y])]):
        node.D2(coordinate)
    C = 0
    for v in N.keys():
        for w in N.keys():
            if v != w and distance(N[v], N[w]) == 1:
                C += 1; N[v].neighbors.add(w); N[w].neighbors.add(v)
    print(f'number of edges - empirical: {C}')
    O = { + 1: sum(1 for node in N.values() if node.degree() == 0 and node.state == + 1),
    - 1: sum(1 for node in N.values() if node.degree() == 0 and node.state == - 1) }
    print(f'off-network nodes: {O}')
    return N, E
# returns the dictionary of nodes
# ==================================================
# the 3D Regular Grid generation function generates the network N (representing the dictionary of network nodes) according to the 3D Regular Grid model,
# with edge probability as a function of the distance between nodes
def D3(n, p):
    def distance(v, w):
        return abs(v.x - w.x) + abs(v.y - w.y) + abs(v.z - w.z)
    print(f'\nsimulating 3D grid with state probability = {p}')
    N = { i: Node(i, numpy.random.choice([+ 1, - 1], size = None, p = [p, 1 - p])) for i in range(n) }
    E = None
    def arange(n, k):
        return numpy.arange(start = 0, stop = math.pow(n, 1 / k))
    X, Y, Z = arange(n, 3), arange(n, 3), arange(n, 3)
    for node, coordinate in zip(N.values(), [c for c in itertools.product(* [X, Y, Z])]):
        node.D3(coordinate)
    C = 0
    for v in N.keys():
        for w in N.keys():
            if v != w and distance(N[v], N[w]) == 1:
                C += 1; N[v].neighbors.add(w); N[w].neighbors.add(v)
    print(f'number of edges - empirical: {C}')
    O = { + 1: sum(1 for node in N.values() if node.degree() == 0 and node.state == + 1),
    - 1: sum(1 for node in N.values() if node.degree() == 0 and node.state == - 1) }
    print(f'off-network nodes: {O}')
    return N, E
# returns the dictionary of nodes
# ==================================================
# the simulation function generates the network according to the model,
# set the FES and iterate until unitary consensus
def simulation (n, p, m):
    N, E = m(n, p)
    time = 0
    status = {+ 1: round(sum(1 for node in N.values() if node.state == + 1) / numpy.array([node for node in N.keys()]).size * 100, 2),
    - 1: round(sum(1 for node in N.values() if node.state == - 1) / numpy.array([node for node in N.keys()]).size * 100, 2)}
    print(f'START - time: {round(time, 2)}, status: {status}')
    start = (copy.deepcopy(N), copy.deepcopy(E), time)
    FES = queue.PriorityQueue()
    FES.put((0, numpy.random.choice([node for node in N.keys()], size = None)))
    while sum(1 for node in N.values() if node.state == + 1) < numpy.array([node for node in N.keys()]).size:
        (time, node) = FES.get()
        wakeup(time, FES, N, node)
        k = sum(1 for node in N.values() if node.state == + 1)
        print(end = '\r|%-100s|' % ('-' * (100 * (k) // numpy.array([node for node in N.keys()]).size)))
        if sum(1 for node in N.values() if node.state == - 1) == numpy.array([node for node in N.keys()]).size: break
    status = {+ 1: round(sum(1 for node in N.values() if node.state == + 1) / numpy.array([node for node in N.keys()]).size * 100, 2),
    - 1: round(sum(1 for node in N.values() if node.state == - 1) / numpy.array([node for node in N.keys()]).size * 100, 2)}
    C = 0 if status[+ 1] < 100 else 1
    print(f'consensus: {C}')
    end = (N, E, time)
    print(f'\nEND - time: {round(time, 2)}, status: {status}')
    return time, start, end, N, C
# returns the simulation time, the initial state of the network, the final state of the network,
# the dictionary of nodes and the final consensus value (+1 or -1)
# ==================================================
# the output analysis function, with correction for one-observation population cases
def output (X, CL):
    if numpy.array(X).size < 2:
        return {'MV': numpy.mean(X), 'STD': 0, 'A': 0, 'I': 0}
    DF = 1
    MV = numpy.mean(X)
    STD = math.sqrt(numpy.var(X, ddof = DF))
    SEM = STD / math.sqrt(numpy.array(X).size)
    I = scipy.stats.t.interval(CL, DF, MV, SEM)   
    EA = (max(I) - min(I)) / 2
    ER = EA / MV
    A = 1 - ER if 1 - ER > 0 else 0
    return {'MV': MV, 'STD': STD, 'A': A, 'I': I}
# returns the mean value, the standard deviation, the accuracy and the confidence interval
# ==================================================
# the drawing function illustrates the initial and final states of the network and the distribution of the network's degrees of freedom
def draw(times, consensus, degrees, start, end, m, p):
    font = { 'fontname': 'Times New Roman', 'fontsize': '12.5', 'color': 'black' }
    colors = { + 1: 'crimson', - 1: 'navy'}
    if m is G:

        for state, name in zip([start, end], ['S', 'E']):
            (N, E, time) = state
            colormap = [colors[node.state] for node in N.values()]
            graph = networkx.Graph()
            network = [node for node in N.keys()]
            graph.add_nodes_from(network)
            for (v, w) in E:
                if v != w:
                    if v in network and w in network:
                        graph.add_edge(v,w)
            networkx.draw(graph, nodelist = N.keys(), node_size = 10, width = 0.05, node_color = colormap, alpha = 1)
            plot.show()
        x = numpy.random.poisson(n * 10 / n, size = n)
        y = numpy.concatenate([array for array in degrees], axis = None)
        plot.hist([x, y], label = ['theoretical', 'empirical'], color = ['navy', 'crimson'], bins = 50, density = True)
        plot.title(f'average time: {round(output(times, CL)["MV"], 2)}, consensus probability: {round(output(consensus, CL)["MV"] * 100, 2)}%')
        plot.legend()
        plot.show()

    if m is D2:
        for state, name in zip([start, end], ['S', 'E']):
            (N, E, time) = state
            colormap = [colors[node.state] for node in N.values()]
            X = [node.x for node in N.values()]
            Y = [node.y for node in N.values()]
            plot.scatter(X, Y, c = colormap, s = 10, alpha = 0.5)
            for v in N.keys():
                for w in N[v].neighbors:
                    plot.plot([N[v].x, N[w].x], [N[v].y, N[w].y], c = 'black', linewidth = 0.15, alpha = 1)
            plot.grid(False); plot.axis('off')
            plot.tight_layout()
            plot.show()
        plot.hist(numpy.concatenate([array for array in degrees], axis = None), bins = 50, density = True, color = 'crimson')
        plot.title(f'average time: {round(output(times, CL)["MV"], 2)}, consensus probability: {round(output(consensus, CL)["MV"] * 100, 2)}%')
        plot.show()

    if m is D3:
        for state, name in zip([start, end], ['S', 'E']):
            (N, E, time) = state
            colormap = [colors[node.state] for node in N.values()]
            X = [node.x for node in N.values()]
            Y = [node.y for node in N.values()]
            Z = [node.z for node in N.values()]
            ax = plot.axes(projection = '3d')
            ax.scatter(X, Y, Z, c = colormap, s = 10, alpha = 0.5)
            for v in N.keys():
                for w in N[v].neighbors:
                    ax.plot([N[v].x, N[w].x], [N[v].y, N[w].y], [N[v].z, N[w].z], c = 'black', linewidth = 0.15, alpha = 1)
            ax.grid(False); plot.axis('off')
            plot.tight_layout()
            plot.show()
        plot.hist(numpy.concatenate([array for array in degrees], axis = None), bins = 50, density = True, color = 'crimson')
        plot.title(f'average time: {round(output(times, CL)["MV"], 2)}, consensus probability: {round(output(consensus, CL)["MV"] * 100, 2)}%')
        plot.show()
# ==================================================
models = { 'G': G, 'D2': D2, 'D3': D3 }
# ==================================================
# consider a voter model over a G(n,p) with n chosen in the range [10^3, 10^4] and p_g= 10/n,
# according to the initial condition, each node  has a probability p_1  of being in state +1 with p_1\in {0.51, 0.55, 0.6, 0.7},
# evaluate the probability of reaching  a + 1 consensus (if the graph is not connected consider only the giant component),
# evaluate, as well, the time needed to reach consensus.
CL = 0.80; A = 0.80
n = 1000
for k, m in models.items():
    if m is G:
        for p in { 0.51, 0.55, 0.6, 0.7 }:
            TIMES, STARTS, ENDS, CONSENSUS, DEGREES, ITER = [], [], [], [], [], 0
            # while min(output(CONSENSUS, CL)["A"], output(TIMES, CL)["A"]) < A:
            for i in range(1):
                time, start, end, N, C = simulation(n, p, m)
                TIMES.append(time), CONSENSUS.append(C)
                DEGREES.append(numpy.array([node.degree() for node in N.values()]))
                STARTS.append(start), ENDS.append(end)
                print(f'accuracy: {round(min(output(CONSENSUS, CL)["A"], output(TIMES, CL)["A"]) * 100, 2)}, iter. {ITER + 1}'); ITER += 1
            index = CONSENSUS.index(collections.Counter(CONSENSUS).most_common(1)[0][0])
            draw(TIMES, CONSENSUS, DEGREES, STARTS[index], ENDS[index], m, p)
            print(f'\nRESULT - average time: {round(output(TIMES, CL)["MV"], 2)}, consensus probability: {round(output(CONSENSUS, CL)["MV"] * 100, 2)}%\n')
# ==================================================
# consider a voter model over finite portion of Z^2 and Z^3,
# fix p_1=0.51 and, for 2/3 values of n \in[10^2, 10^4], estimate, as before,
# the probability of reaching  a + 1 consensus  and the time needed to reach consensus.
CL = 0.80; A = 0.80
n = 1000
for k, m in models.items():
    if m is not G:
        for p in { 0.51 }:
            TIMES, STARTS, ENDS, CONSENSUS, DEGREES, ITER = [], [], [], [], [], 0
            # while min(output(CONSENSUS, CL)["A"], output(TIMES, CL)["A"]) < A:
            for i in range(1):
                time, start, end, N, C = simulation(n, p, m)
                TIMES.append(time), CONSENSUS.append(C)
                DEGREES.append(numpy.array([node.degree() for node in N.values()]))
                STARTS.append(start), ENDS.append(end)
                print(f'accuracy: {round(min(output(CONSENSUS, CL)["A"], output(TIMES, CL)["A"]) * 100, 2)}, iter. {ITER + 1}'); ITER += 1
            index = CONSENSUS.index(collections.Counter(CONSENSUS).most_common(1)[0][0])
            draw(TIMES, CONSENSUS, DEGREES, STARTS[index], ENDS[index], m, p)
            print(f'\nRESULT - average time: {round(output(TIMES, CL)["MV"], 2)}, consensus probability: {round(output(CONSENSUS, CL)["MV"] * 100, 2)}%\n')