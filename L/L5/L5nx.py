import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple
from scipy import stats
from math import comb
from tqdm import tqdm   # progress bar

# constants for the considered models
LAMBDA_V  = 1
LAMBDA_VW = 1

def graph_erdosrenyi(n, p):
    graph = nx.Graph()
    nodes_temp = np.arange(0,n)
    pbar = tqdm(range(n-1), ncols=120, desc='Generating random graph (networkx)')
    for i in pbar:
        # generate the nodes to be connected to the current node
        connect_nodes = np.random.choice([True,False], size=n-1-i, p=[p,1-p])
        neighbors = nodes_temp[i+1:][ connect_nodes ]
        graph.add_edges_from((i, j) for j in neighbors)
    return graph

# QQ plot
def degrees_distribution_analysis(graph, graph_n, graph_p):
    degrees = np.array(graph.degree)[:,1]
    n = graph.number_of_nodes()
    poisson_np = graph_n*graph_p
    
    quantiles_empiric = np.sort(degrees)    
    quantiles_theory = np.zeros(n)
    for i in range(n):
        quantiles_theory[i] = stats.poisson.ppf(i/n, poisson_np)

    plt.plot(quantiles_theory, quantiles_empiric, 'k.', markersize=9, alpha=0.1,)
    plt.axline((quantiles_theory[0], quantiles_theory[0]), slope=1, alpha=0.5, linewidth=1, c='r')
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Empirical quantiles')
    plt.xlim([0,max(quantiles_theory)+1])
    plt.ylim([0,max(quantiles_empiric)+1])
    xticks=np.arange( 0, max(quantiles_theory)+2, max(1, max(quantiles_theory)//10) ) 
    plt.xticks(xticks)
    yticks=np.arange( 0, max(quantiles_empiric)+2, max(1, max(quantiles_empiric)//10) ) 
    plt.yticks(yticks)
    plt.grid(alpha=0.75)
    plt.show()

class Event:
    # Since the events are basically wake-up a node/edge
    # the FES is handled with a single next_event
    def __init__(self, time, node=None, edge=None):
        # allow handling events for either nodes or edges
        self.time=time
        self.node=node
        self.edge=edge
    def __repr__(self):
        s = f'Event: {self.time:.5f} - '
        s+= f'Node: {self.node}' if self.node else ''
        s+= f'Edge: {self.edge}' if self.edge else ''
        return s

def stop_condition(wakeup_total_count, nodes_at_min_wakeup, total_nodes,
                   min_wakeup_condition=None, min_wakeup_per_node_condition=None):
    # Return True if any stop condition given is satisfied, False otherwise
    stop_sim = False
    if min_wakeup_condition:
        stop_sim |= (wakeup_total_count == min_wakeup_condition)
    if min_wakeup_per_node_condition:
        stop_sim |= (nodes_at_min_wakeup == total_nodes)
    return stop_sim

def averaging_model(graph, alpha, initial_distribution, min_wakeup_condition=None, 
                    min_wakeup_per_node_condition=None, verbose=True):
    """
    Simulate an averaging model dynamic graph
    Args:
        graph : input nodes and edges ;
        alpha : averaging model parameter ;
        initial_distribution (dict): contains the distribution ('uniform' or 'beta')
            name and its parameters ;
        min_wakeup_condition (int): simulation stops when this number of events are concluded ;
        min_wakeup_per_node_condition (int): simulation stops when all nodes wake up at least
            this amount of times ;
    Returns:
        graph: final state of the graph
    """    
    graph = graph.copy()
    n = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if not any([min_wakeup_condition, min_wakeup_per_node_condition]):
        # at least one stop condition must be given - default case
        min_wakeup_per_node_condition = 1
    
    wakeup_total_count = 0
    min_wakeup_per_node = 0
    nodes_at_min_wakeup = 0
    # consider property of poisson processes - any edge wake up with rate lambda_n
    lambda_n = LAMBDA_VW * n_edges

    # initialization
    if initial_distribution['dist'] == 'uniform':
        for v in graph.nodes:
            graph.nodes[v]['state'] = np.random.uniform(0,1)
            graph.nodes[v]['wakeup_count'] = 0
    elif initial_distribution['dist'] == 'beta':
        a = initial_distribution['a']
        b = initial_distribution['b']
        for v in graph.nodes:
            graph.nodes[v]['state'] = np.random.beta(a,b)
            graph.nodes[v]['wakeup_count'] = 0

    # first event
    v = np.random.choice( graph.nodes )
    w = np.random.choice( list(graph.neighbors(v)) )
    event = Event( time=0, edge=(v,w) )
    next_event = event
    clock=0

    print('\nAVERAGING MODEL')
    stop_sim = False
    # while wakeup_total_count < min_wakeup_condition:
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        clock, (v,w) = event.time, event.edge
        # get current state and update nodes state
        v_minus, w_minus = graph.nodes[v]['state'], graph.nodes[w]['state']
        graph.nodes[v]['state'] = alpha*v_minus + (1-alpha)*w_minus
        graph.nodes[w]['state'] = alpha*w_minus + (1-alpha)*v_minus

        graph.nodes[v]['wakeup_count'] += 1
        if graph.nodes[v]['wakeup_count'] == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        graph.nodes[w]['wakeup_count'] += 1
        if graph.nodes[w]['wakeup_count'] == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        
        next_wakeup_time = clock + np.random.exponential( scale=1/lambda_n )
        next_v = np.random.choice( graph.nodes )
        next_w = np.random.choice( list(graph.neighbors(next_v)) )
        next_event = Event(time=next_wakeup_time, edge=(next_v, next_w))

        stop_sim = stop_condition(
            wakeup_total_count, 
            nodes_at_min_wakeup, 
            total_nodes=n,
            min_wakeup_condition=min_wakeup_condition, 
            min_wakeup_per_node_condition=min_wakeup_per_node_condition
        )

        if verbose:
            if min_wakeup_condition:
                print(f'\rNodes wake up: {wakeup_total_count}/{min_wakeup_condition} - Time: {clock:.5f}', end='', flush=True)
            else:
                print(f'\rNodes waken up at least {min_wakeup_per_node_condition} time(s): {nodes_at_min_wakeup}/{n} - Time: {clock:.5f}', end='', flush=True)
    if verbose:
        print(f'\nTotal number of events: {wakeup_total_count}')

    return graph

def linearthreshold_model(graph, r, initial_p0=0.5, min_wakeup_condition=None, 
                          min_wakeup_per_node_condition=None, verbose=True):
    """
    Simulate an linear threshold model dynamic graph
    Args:
        graph : input nodes and edges ;
        r : linear threshold model parameter ;
        initial_p0 : initial probability P(node_state=0).
        min_wakeup_condition (int): simulation stops when this number of events are concluded ;
        min_wakeup_per_node_condition (int): simulation stops when all nodes wake up at least
            this amount of times ;
    Returns:
        graph: final state of the graph
    """    
    graph = graph.copy()
    n = graph.number_of_nodes()

    if not any([min_wakeup_condition, min_wakeup_per_node_condition]):
        # at least one stop condition must be given - default case
        min_wakeup_per_node_condition = 1
    
    wakeup_total_count = 0
    min_wakeup_per_node = 0
    nodes_at_min_wakeup = 0
    # consider property of poisson processes - any node wake up with rate lambda_n
    lambda_n = LAMBDA_V * n

    # initialization
    for v in graph.nodes:
        graph.nodes[v]['state'] = 0 if np.random.uniform(0,1)<initial_p0 else +1
        graph.nodes[v]['wakeup_count'] = 0
    
    # first event
    node = np.random.choice( graph.nodes )
    event = Event( time=0, node=node )
    next_event = event
    clock=0

    print('\nLINEAR THRESHOLD MODEL')
    stop_sim = False
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        clock, v = event.time, event.node
        # get current state from node neighbors
        neighbors_state = np.fromiter((graph.nodes[nbor]['state'] for nbor in graph.neighbors(v)), dtype='?')
        Nv = np.sum(neighbors_state)
        graph.nodes[v]['state'] = 1 if Nv>r else 0
        # next event
        next_wakeup_time = clock + np.random.exponential( scale=1/lambda_n )
        next_wakeup_node = np.random.choice(graph.nodes)
        next_event = Event(time=next_wakeup_time, node=next_wakeup_node)

        graph.nodes[v]['wakeup_count'] += 1
        if graph.nodes[v]['wakeup_count'] == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        
        stop_sim = stop_condition(
            wakeup_total_count, 
            nodes_at_min_wakeup, 
            total_nodes=n,
            min_wakeup_condition=min_wakeup_condition, 
            min_wakeup_per_node_condition=min_wakeup_per_node_condition
        )

        if verbose:
            if min_wakeup_condition:
                print(f'\rNodes wake up: {wakeup_total_count}/{min_wakeup_condition} - Time: {clock:.5f}', end='', flush=True)
            else:
                print(f'\rNodes waken up at least {min_wakeup_per_node_condition} time(s): {nodes_at_min_wakeup}/{n} - Time: {clock:.5f}', end='', flush=True)
    if verbose:
        print(f'\nTotal number of events: {wakeup_total_count}')

    return graph

if __name__ == '__main__':
    np.random.seed(40028922)

    n = 5_000
    p = 10/n
    graph = graph_erdosrenyi(n, p)
    print(graph)

    degrees_distribution_analysis(graph, n, p)

    alpha = 1/3
    initial_distribution={'dist':'uniform'}
    # initial_distribution={'dist':'beta', 'a':2, 'b':5}
    averaging_output = averaging_model(
                            graph, 
                            alpha, 
                            initial_distribution=initial_distribution, 
                            min_wakeup_per_node_condition=1
                        )    
    r = 7
    linearthreshold_output = linearthreshold_model(
                                graph, 
                                r, 
                                initial_p0=0.65, 
                                min_wakeup_per_node_condition=1                       
                             )