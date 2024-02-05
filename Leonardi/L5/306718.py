import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import stats

# constants for the considered models
LAMBDA_V  = 1
LAMBDA_VW = 1

class Node:
    # store node attributes
    id_ = itertools.count()
    def __init__(self):
        self.id = next(Node.id_)
        self.degree = 0
        self.state = None
        self.wakeup_count = 0
    def __repr__(self):
        return f'{self.id}'
    def __str__(self):
        return f'Node {self.id}'
    def __hash__(self):
        return self.id

class Graph:
    def __init__(self, n, p):
        # generates a graph G(n,p)
        self.n = n
        self.p = p
        self.nodes = np.array([ Node() for _ in range(n) ])
        self.edges = set()
        self.neighbors = { node:set() for node in self.nodes }
        
        self.generate_edges() # Erdos-Renyi Model
    
        # filter isolated nodes
        self.nodes = np.array([ node for node in self.nodes if node.degree>0 ])
        self.number_of_nodes = len(self.nodes)
        self.number_of_edges = len(self.edges)

    def __repr__(self):
        return f'Graph - {self.number_of_nodes} nodes - {self.number_of_edges} edges.'

    def generate_edges(self):
        # applies Erdos Renyi model for generating the graph
        for i,node in enumerate(self.nodes[:-1]):
            n,p = self.n, self.p
            u = np.random.uniform(0,1, size=n-1-i)
            neighbors = self.nodes[i+1:][ u<p ] # succeeding nodes to be connected
            self.make_neighbors(node, neighbors)
            print(f'\rGenerating random graph: {int(100*(i+1)/n)}% ({i+1}/{n} nodes)', end='', flush=True)
        print(f'\rGenerating random graph: 100% ({n}/{n} nodes)', flush=True)
    
    def make_neighbors(self, node, neighbors):
        self.neighbors[node] = self.neighbors[node].union(neighbors)
        node.degree += len(neighbors)
        for nbor in neighbors:
            self.edges.add((node,nbor))        
            self.neighbors[nbor].add(node)
            nbor.degree += 1

    def random_node(self):
        return np.random.choice( self.nodes )
    def random_neighbor(self, node):
        neighbors = list( self.neighbors[node] )
        return np.random.choice( neighbors )

def degrees_distribution_analysis(graph, graph_n, graph_p, conf_level=0.95):
    # graph_n and graph_p were were (n,p) to generate the graph
    poisson_np = graph_n*graph_p
    degrees = [ node.degree for node in graph.nodes ]
    # number of non isolated nodes on the graph
    n = graph.number_of_nodes
    
    # get empiric and theoretical distribution of nodes' degrees
    quantiles_empiric = np.sort(degrees)    
    quantiles_theory = np.zeros(n)
    # skip first since for large n, for the first quantile is already zero
    for i in range(1,n):
        quantiles_theory[i] = stats.poisson.ppf(i/n, poisson_np)

    # QQ plot
    plt.plot(quantiles_theory, quantiles_empiric, 'k.', markersize=9, alpha=0.25,)
    plt.axline((1,1), slope=1, alpha=0.5, linewidth=1, c='r')
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

def stop_condition(wakeup_total_count, nodes_at_min_wakeup, total_nodes,
                   min_wakeup_condition=None, min_wakeup_per_node_condition=None):
    # Return True if any stop condition given is satisfied, False otherwise
    stop_sim = False
    if min_wakeup_condition:
        stop_sim |= (wakeup_total_count == min_wakeup_condition)
    if min_wakeup_per_node_condition:
        stop_sim |= (nodes_at_min_wakeup == total_nodes)
    return stop_sim

def linearthreshold_model(graph, r, initial_p1=0.5, min_wakeup_condition=None, 
                          min_wakeup_per_node_condition=None, verbose=False):
    """
    Simulate an linear threshold model dynamic graph
    Args:
        graph : input nodes and edges ;
        r : linear threshold model parameter ;
        initial_p1 : initial probability P(node_state=1).
        min_wakeup_condition (int): simulation stops when this number of events are concluded ;
        min_wakeup_per_node_condition (int): simulation stops when all nodes wake up at least
            this amount of times ;
    Returns:
        initial states: initial state of the nodes
        final states: final state of the nodes
    """    
    n = graph.number_of_nodes

    if not any([min_wakeup_condition, min_wakeup_per_node_condition]):
        # at least one stop condition must be given - default case
        min_wakeup_per_node_condition = 1
    
    # variables to track stop condition
    wakeup_total_count = 0
    nodes_at_min_wakeup = 0
    # property of poisson processes - any edge wake up with rate lambda_n
    lambda_n = LAMBDA_V * n

    # initialization
    for v in graph.nodes:
        v.state = 1 if np.random.uniform(0,1)<initial_p1 else 0
        v.wakeup_count = 0
    # first event
    node = graph.random_node()
    event = Event( time=0, node=node )
    next_event = event
    clock=0

    initial_states = np.array([v.state for v in graph.nodes])
    if verbose:
        print('\n--- LINEAR THRESHOLD MODEL ---')
        print(f'Initial average state: {initial_states.mean():.6f}')
    stop_sim = False
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        clock, v = event.time, event.node
        # get current state from node neighbors
        neighbors_state = np.array([ nbor.state for nbor in graph.neighbors[v] ])
        Nv = np.sum(neighbors_state)
        v.state = 1 if Nv>r else 0

        # re-evaluates the stop condition
        v.wakeup_count += 1
        if v.wakeup_count == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        stop_sim = stop_condition(
            wakeup_total_count, 
            nodes_at_min_wakeup, 
            total_nodes=n,
            min_wakeup_condition=min_wakeup_condition, 
            min_wakeup_per_node_condition=min_wakeup_per_node_condition
        )

        # next event
        next_wakeup_time = clock + np.random.exponential( scale=1/lambda_n )
        next_wakeup_node = graph.random_node()
        next_event = Event(time=next_wakeup_time, node=next_wakeup_node)

        if verbose:
            if min_wakeup_condition:
                print(f'\rNodes wake up: {wakeup_total_count}/{min_wakeup_condition} ({(wakeup_total_count/min_wakeup_condition):.1%})', end='', flush=True)
            else:
                print(f'\rNodes waken up at least {min_wakeup_per_node_condition} time(s): {nodes_at_min_wakeup}/{n} ({(nodes_at_min_wakeup/n):.1%})', end='', flush=True)

    final_states = np.array([v.state for v in graph.nodes])
    if verbose:
        print(f'\nTotal number of events: {wakeup_total_count}')
        print(f'Final average state: {final_states.mean():.6f}')
    return initial_states, final_states

def averaging_model(graph, alpha, initial_distribution, min_wakeup_condition=None, 
                    min_wakeup_per_node_condition=None, verbose=False):
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
        initial states: initial state of the nodes
        final states: final state of the nodes
    """    
    n = graph.number_of_nodes
    m = graph.number_of_edges

    if not any([min_wakeup_condition, min_wakeup_per_node_condition]):
        # at least one stop condition must be given - default case
        min_wakeup_condition = n
    
    # variables to track stop condition
    wakeup_total_count = 0
    nodes_at_min_wakeup = 0
    # property of poisson processes - any edge wake up with rate lambda_m
    lambda_m = LAMBDA_VW * m

    # initialization
    if initial_distribution['dist'] == 'uniform':
        for v in graph.nodes:
            v.state = np.random.uniform(0,1)
            v.wakeup_count = 0
    elif initial_distribution['dist'] == 'beta':
        a = initial_distribution['a']
        b = initial_distribution['b']
        for v in graph.nodes:
            v.state = np.random.beta(a,b)
            v.wakeup_count = 0

    # first event
    v = graph.random_node()
    w = graph.random_neighbor(v)
    event = Event( time=0, edge=(v,w) )
    next_event = event
    clock=0

    initial_states = np.array([v.state for v in graph.nodes])
    if verbose:
        print('\n--- AVERAGING MODEL ---')
        print(f'Initial average state: {initial_states.mean():.6f}')
    stop_sim = False
    # while wakeup_total_count < min_wakeup_condition:
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        clock, (v,w) = event.time, event.edge
        # get current state and update nodes state
        v_minus, w_minus = v.state, w.state
        v.state = alpha*v_minus + (1-alpha)*w_minus
        w.state = alpha*w_minus + (1-alpha)*v_minus

        # re-evaluates the stop condition
        v.wakeup_count += 1
        if v.wakeup_count == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        w.wakeup_count += 1
        if w.wakeup_count == min_wakeup_per_node_condition:
            nodes_at_min_wakeup += 1
        stop_sim = stop_condition(
            wakeup_total_count, 
            nodes_at_min_wakeup, 
            total_nodes=n,
            min_wakeup_condition=min_wakeup_condition, 
            min_wakeup_per_node_condition=min_wakeup_per_node_condition
        )
        
        # get next event
        next_wakeup_time = clock + np.random.exponential( scale=1/lambda_m )
        next_v = graph.random_node()
        next_w = graph.random_neighbor(v)
        next_event = Event(time=next_wakeup_time, edge=(next_v, next_w))

        if verbose:
            if min_wakeup_condition:
                print(f'\rNodes wake up: {wakeup_total_count}/{min_wakeup_condition} ({(wakeup_total_count/min_wakeup_condition):.1%})', end='', flush=True)
            else:
                print(f'\rNodes waken up at least {min_wakeup_per_node_condition} time(s): {nodes_at_min_wakeup}/{n} ({(nodes_at_min_wakeup/n):.1%})', end='', flush=True)
    
    final_states = np.array([v.state for v in graph.nodes])
    if verbose:
        print(f'\nTotal number of events: {wakeup_total_count}')
        print(f'Final average state: {final_states.mean():.6f}')
    return initial_states, final_states

def plot_averaging_model(averaging_model_outputs, parameters):
    # plot distributions before and after simulation of averaging model
    # for a list of different parameters
    count = len(averaging_model_outputs)
    fig, ax = plt.subplots(count, sharex=True, figsize=(5,9))
    fig.tight_layout()
    for i,output in enumerate(averaging_model_outputs):
        alpha, initial_dist = parameters[i]
        initial_state, final_state = output
        ax[i].hist(initial_state, bins=np.arange(0,1.01,0.025), alpha=2/3, label='Initial state')
        ax[i].hist(final_state, bins=np.arange(0,1.01,0.025), alpha=2/3, label='Final state')
        ax[i].set_xlim(0,1)
        ax[i].set_xticks(np.arange(0,1.01,.1))
        ax[i].grid()
        ax[i].set_title(f'distribution: {initial_dist['dist']} - alpha={alpha}', fontsize=9)
    ax[0].legend(fontsize=9)
    plt.show()

def plot_linearthreshold_model(linearthreshold_final_states, r_params, initial_p1_params):
    # plot final probability of node state equal 1 for different inputs in the linear threshold method
    n1, n2 = len(r_params), len(initial_p1_params)
    # transpose and flip the matrix for the heat map
    values = np.transpose(linearthreshold_final_states)
    values = np.flip(values, axis=0)
    fig,ax = plt.subplots()
    heatmap = ax.imshow(values, vmin=0,vmax=1, extent=[0, n1, 0, n2], cmap='plasma')
    for i in range(n1):
        for j in range(n2):
            value = values[n1-(i+1)][j]
            color = 'white' if value<.5 else 'black'
            ax.annotate(f'{value:.2f}', xy=(j+0.5, i+0.5), ha='center', va='center', color=color, fontsize=9)
    plt.colorbar(heatmap, ax=ax)

    ax.set_xticks(np.arange(n2)+0.5)
    ax.set_yticks(np.arange(n2)+0.5)
    ax.set_xticklabels([f'{r:.0f}' for r in r_params])
    ax.set_yticklabels([f'{p1:.1f}' for p1 in initial_p1_params])
    ax.set_xlabel("r parameter")
    ax.set_ylabel("initial P(X_v=1)")
    plt.show()

if __name__ == '__main__':
    np.random.seed(40028922)

    # for n=100k, total run time around 5 minutes
    n = 10_000
    n = 100_000
    p = 10/n
    print(f'--- GRAPH G(n={n:.0e}, p={p:.0e}) ---')
    graph = Graph(n,p)
    print(graph)

    degrees_distribution_analysis(graph, n, p)

    averaging_model_outputs = []
    parameters = [
        (1/2, {'dist':'uniform'}),
        (1/4, {'dist':'uniform'}),
        (1/4, {'dist':'beta', 'a':4.0, 'b':1.5}),
        (1/4, {'dist':'beta', 'a':1.5, 'b':4.0}),
    ]
    print('\n--- AVERAGING MODEL ---')
    for alpha, initdist in parameters:
        print(f'\rinitial distribution {initdist['dist']} - alpha={alpha:.2f}', end='    ')
        avg_initial_state, avg_final_state = averaging_model(
                                            graph, 
                                            alpha, 
                                            initial_distribution=initdist, 
                                            min_wakeup_condition=n,
                                            verbose=False
                                            # min_wakeup_per_node_condition=1,
                                        )
        averaging_model_outputs.append((avg_initial_state, avg_final_state))
    plot_averaging_model(averaging_model_outputs, parameters)
    
    r_param = np.arange(1,10)
    initial_p1_param = np.arange(0.1,1,0.1)
    n1,n2 = len(r_param), len(initial_p1_param)
    lth_final_states = np.zeros((n1,n2))
    print('\n--- LINEAR THRESHOLD MODEL ---')
    for i,r in enumerate(r_param):
        for j,initial_p1 in enumerate(initial_p1_param):
            print(f'\rr={r} - initial_p1={initial_p1:.2f}', end='')
            lth_initial_state, lth_final_state = linearthreshold_model(graph, r, initial_p1, 
                                                                       min_wakeup_condition=n,
                                                                       # min_wakeup_per_node_condition=1
                                                                    )
            lth_final_states[i,j] = lth_final_state.mean()
    plot_linearthreshold_model(lth_final_states, r_param, initial_p1_param)