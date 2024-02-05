import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from collections import namedtuple, defaultdict
from tqdm import tqdm

# constants for the considered model
LAMBDA_V  = 1

class Node:
    # store node attributes
    id_ = itertools.count()
    def __init__(self, coordinates:tuple=None):
        self.id = next(Node.id_)
        self.degree = 0
        self.state = 0
        self.wakeup_count = 0
        self.coordinates = coordinates # given when handling grid graph
    def __repr__(self):
        if isinstance(self.coordinates, tuple):
            return f'Node({self.coordinates})'
        return f'Node({self.id})'
    def __str__(self):
        if isinstance(self.coordinates, tuple):
            return f'Node({self.coordinates}, state={self.state}, degree={self.degree})'
        return f'Node({self.id}, state={self.state}, degree={self.degree})'
    def __hash__(self):
        return self.id

class Graph:
    def __init__(self, n:int=None, p:float=None, sizes:int=None, check_giant_component:bool=True):
        self.n = n
        self.p = p
        self.sizes = sizes
        self.edges = set()
        self.neighbors = defaultdict(set)
        self.generate_edges() 
        
        if check_giant_component:
            # consider only largest component if graph is not connected
            self.get_giant_component() 

    def __repr__(self):
        return f'Graph(nodes={self.number_of_nodes()}; edges={self.number_of_edges()})'

    def generate_edges(self):
        if self.n and self.p:
            # apply Erdos Renyi model for generating the graph G(n,p)
            self.nodes = np.fromiter( (Node() for _ in range(self.n)), Node, self.n )
            for i,node in enumerate(self.nodes[:-1]):
                n,p = self.n, self.p
                u = np.random.uniform(0,1, size=n-1-i)
                neighbors = self.nodes[i+1:][ u<p ] # succeeding nodes to be connected
                self.make_neighbors(node, neighbors)
            # filter isolated nodes
            self.nodes = np.fromiter( filter(lambda v:v.degree!=0, self.nodes), Node )
        else:
            # generate regular grid graph
            Z = len(self.sizes)
            nodes_coord = itertools.product(*(np.arange(size) for size in self.sizes))
            # instantiate a node a each coordinate of the grid
            self.nodes = np.fromiter( (Node(coordinates=coord) for coord in nodes_coord), Node, count=np.prod(self.sizes) )
            self.nodes_grid = self.nodes.reshape(self.sizes)
            # connect neighbor nodes
            for i in range(self.sizes[0]):
                for j in range(self.sizes[1]):
                    if Z==2:
                        v = self.nodes_grid[i,j]
                        v_neighbors_list = []
                        if i>0:
                            v_neighbors_list.append(self.nodes_grid[i-1,j])
                        if j>0:
                            v_neighbors_list.append(self.nodes_grid[i,j-1])
                        self.make_neighbors(v, v_neighbors_list)
                    elif Z == 3:
                        for k in range(self.sizes[2]):
                            v = self.nodes_grid[i,j,k]
                            v_neighbors_list = []
                            if i>0:
                                v_neighbors_list.append(self.nodes_grid[i-1,j,k])
                            if j>0:
                                v_neighbors_list.append(self.nodes_grid[i,j-1,k])
                            if k>0:
                                v_neighbors_list.append(self.nodes_grid[i,j,k-1])
                            self.make_neighbors(v, v_neighbors_list)


    def make_neighbors(self, node:Node, neighbors_list:list):
        self.neighbors[node] = self.neighbors[node].union(neighbors_list)
        node.degree += len(neighbors_list)
        for nbor in neighbors_list:
            self.edges.add((node,nbor))        
            self.neighbors[nbor].add(node)
            nbor.degree += 1
    
    def number_of_nodes(self):
        return len(self.nodes)
    def number_of_edges(self):
        return len(self.edges)

    def get_giant_component(self):
        # if graph is not connected, consider only largest component
        components = []
        for (v,w) in self.edges:
            v_compo, w_compo = -1, -1
            # check if v and w are already in any component
            for i,component in enumerate(components):
                if v in component:
                    v_compo = i
                if w in component:
                    w_compo = i
            if v_compo==w_compo!=-1:
                pass # both nodes already on the same component
            elif v_compo==-1 and w_compo==-1:
                # first appearance of both nodes - new component
                components.append( {v,w} ) 
            elif v_compo==-1:
                # first appearance of v - add it to w component
                components[w_compo].add(v)
            elif w_compo==-1:
                # first appearance of w - add it to v component
                components[v_compo].add(w)
            else:
                # v and w in different components - join them
                components[v_compo] = components[v_compo].union( components[w_compo] )
                components.pop(w_compo)
        
        if len(components) > 1:
            print('One giant component considered', end=' ')
            giant_component = max(components, key=len)
            n,m = self.number_of_nodes(), self.number_of_edges()
            # update graph elements to consider only giant component
            self.nodes = np.fromiter(giant_component, Node, len(giant_component))
            self.edges = set( filter(lambda e:{e[0],e[1]}.issubset(giant_component), self.edges) )
            self.neighbors = dict( filter(lambda kv:kv[0] in giant_component, self.neighbors.items()) )        
            print(f'({n-self.number_of_nodes()} nodes and {m-self.number_of_edges()} edges discarded)')

    def state(self) -> np.ndarray:
        # return array with state of nodes
        return np.fromiter(
                   iter=(v.state for v in graph.nodes), 
                   dtype=type( graph.nodes[0].state ), 
                   count=self.number_of_nodes()
               )
    def consensus(self) -> bool:
        # return True if the nodes reached consensus
        return len(np.unique(self.state()))==1

    def random_node(self) -> Node:
        return np.random.choice( self.nodes )
    def random_neighbor(self, node:Node) -> Node:
        neighbors = list( self.neighbors[node] )
        return np.random.choice( neighbors )
    
# Since the only event to handle is a node wake-up, the FES can be handled 
# with a single next_event (assuming the nodes wake up at the same rate)
Event = namedtuple('Event', ['time','node'])

def stop_condition(graph, wakeup_total_count, nodes_at_min_wakeup, total_nodes,
                   min_wakeup_condition=None, min_wakeup_per_node_condition=None):
    # Return True if any stop condition given is satisfied, False otherwise
    stop_sim = graph.consensus()
    if min_wakeup_condition:
        stop_sim |= (wakeup_total_count == min_wakeup_condition)
    if min_wakeup_per_node_condition:
        stop_sim |= (nodes_at_min_wakeup == total_nodes)
    return stop_sim

def voter_model(graph:Graph, p1:float=0.51, min_wakeup_condition:int=None, 
                min_wakeup_per_node_condition:int=None) -> float:
    """
    Simulate a voter model dynamic graph
    Args:
        graph : input nodes and edges ;
        p1 : initial probability P(node_state=+1) ;
        min_wakeup_condition : simulation stops after this many events ;
        min_wakeup_per_node_condition : simulation stops after each node 
                                        wakes up this many times.
    Returns:
        graph : final state of graph after simulation
    """
    n = graph.number_of_nodes()
    state_log = {} # pairs clock:P(node_state==1)

    # variables to track stop condition
    wakeup_total_count = 0
    nodes_at_min_wakeup = 0

    # initialization of nodes state for the given p1
    u=np.random.uniform(0,1, size=n)
    for i,v in enumerate(graph.nodes):
        v.state = +1 if u[i]<p1 else -1
        v.wakeup_count = 0

    # first event
    node = graph.random_node()
    event = Event( time=0, node=node )
    next_event = event
    clock=0

    initial_state = graph.state()
    
    # progress bar keep track of simulation
    bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}{postfix}'
    pbar = tqdm(desc='Consensus', total=n, initial=initial_state[initial_state==1].sum(), postfix={'events':0}, 
                bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt}{postfix}')
    
    stop_sim = False    
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        clock, v = event.time, event.node
        
        w = graph.random_neighbor(v)
        pbar.update((w.state-v.state)//2)
        pbar.set_postfix({'events':wakeup_total_count})
        v.state = w.state

        # re-evaluates the stop condition
        v.wakeup_count += 1
        if v.wakeup_count == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        stop_sim = stop_condition(
            graph,
            wakeup_total_count, 
            nodes_at_min_wakeup, 
            total_nodes=n,
            min_wakeup_condition=min_wakeup_condition, 
            min_wakeup_per_node_condition=min_wakeup_per_node_condition
        )

        # next event
        next_wakeup_node = graph.random_node()
        next_wakeup_time = clock + np.random.exponential(scale=1/LAMBDA_V)
        next_event = Event(time=next_wakeup_time, node=next_wakeup_node)

        state = graph.state()
        state_log[clock] = len( state[state==+1] ) / n

    pbar.close()
    final_states = graph.state()
    if graph.consensus():
        print(f'{final_states[0]:+}-consensus reached at instant {clock:.3f} after {wakeup_total_count} events.')
    else:
        print(f'No consensus reached after {wakeup_total_count} events.')

    return graph, state_log

def plot_state_evolution(state_log):
    # plot nodes state along time
    time_log = np.fromiter( state_log.keys(), dtype='f' )
    state_log = np.fromiter( state_log.values(), dtype='f' )
    max_time = time_log.max()

    fig,ax = plt.subplots()
    ax.fill_between(time_log, 0, state_log, step='post', label='+1 nodes', alpha=0.7, color='C1', edgecolor="none")
    ax.fill_between(time_log, state_log, 1, step='post', label='-1 nodes', alpha=0.7, color='C2', edgecolor="none")
    ax.set_xlabel('Time')
    ax.set_ylabel('Opinion (%)')
    ax.set_xlim([0,max_time])
    ax.set_ylim((0,1))
    ax.legend(loc='best')
    ax.grid()
    plt.show()

if __name__ == '__main__':
    np.random.seed(40028922)

    print('--- EXPERIMENT 1 - G(n,p) ---')
    n = 1_000
    # n = 300       # uncomment for faster run
    p_g = 10/n
    print(f'--- GRAPH G(n={n:.0e}, p_g={p_g:.0e}) ---')
    graph = Graph(n=n, p=p_g)

    p1_params = [.51,.55,.60,.70]
    for p1 in p1_params:
        print(f'p1={p1:.0%}')
        graph_sim, state_log = voter_model(graph, p1)
        final_state = np.unique( graph_sim.state(), return_counts=True )
        values, counts = final_state
        if not graph_sim.consensus():
            print(f'Final state: ({values[0]:+} nodes - {counts[0]/counts.sum():.2%}) vs. ({values[1]:+} nodes - {counts[1]/counts.sum():.2%})')
        plot_state_evolution(state_log)

    print('\n--- EXPERIMENT 2 - Regular grids over portions of Z2 and Z3 ---')
    p1 = 0.51
    # different sizes of grids over Z2
    sizes_Z2 = [ (10,10), (32,32), (100,100) ]
    # sizes_Z2 = [ (10,10), (12,12), (15,15) ]       # uncomment for faster run
    for sizes in sizes_Z2:
        print(f'--- GRAPH G(Z2; sizes={sizes}) ---')
        graph = Graph(sizes=sizes)
        print(graph)

        min_wakeup_condition = 500000 if np.prod(sizes)>=1000 else None
        graph_sim, state_log = voter_model(graph, p1, min_wakeup_condition=min_wakeup_condition)
        final_state = np.unique( graph_sim.state(), return_counts=True )
        values, counts = final_state
        if not graph_sim.consensus():
            print(f'Final state: ({values[0]:+} nodes - {counts[0]/counts.sum():.2%}) vs. ({values[1]:+} nodes - {counts[1]/counts.sum():.2%})')
        # plot evolution of simulation
        plot_state_evolution(state_log)

    # different sizes of grids over Z3
    sizes_Z3 = [ (5,5,4), (10,10,10), (21,21,22) ]
    # sizes_Z3 = [ (5,5,4), (6,6,6), (8,8,8) ]       # uncomment for faster run
    for sizes in sizes_Z3:
        print(f'--- GRAPH G(Z3; sizes={sizes}) ---')
        graph = Graph(sizes=sizes)
        print(graph)

        min_wakeup_condition = 500000 if np.prod(sizes)>=1000 else None
        graph_sim, state_log = voter_model(graph, p1, min_wakeup_condition=min_wakeup_condition)
        final_state = np.unique( graph_sim.state(), return_counts=True )
        values, counts = final_state
        if not graph_sim.consensus():
            print(f'Final state: ({values[0]:+} nodes - {counts[0]/counts.sum():.2%}) vs. ({values[1]:+} nodes - {counts[1]/counts.sum():.2%})')
        # plot evolution of simulation
        plot_state_evolution(state_log)    