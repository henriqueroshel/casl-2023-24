##############################################################################################
## The simulation lasts about 10min (because, to get a good aproximation of the probability ##
## of reaching  consensus, many trials have to be done). This time can be decreased by      ##
## changing the n_iterations variable (line 80 of this code)                                ##
##############################################################################################

import random
from time import time
from tqdm import tqdm

# Function to create a G(n,p) graph
def generate_graph(n, p):
    graph = [[] for _ in range(n)]  # Adjacency list representation

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:  # Probability check for edge creation
                graph[i].append(j)
                graph[j].append(i)

    return graph

# Depth-First Search (DFS) to find connected components
def dfs(node, graph, visited, component):
    visited[node] = True
    component.append(node)
    
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(neighbor, graph, visited, component)

# Function to find the giant component in the graph
def find_giant_component(graph):
    n = len(graph)
    visited = [False] * n
    giant_component = []

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, graph, visited, component)
            if len(component) > len(giant_component):
                giant_component = component

    return giant_component

# Generates initial states of the nodes
def get_initial_states(n, p1):
    list_states = [random.choices([+1, 0], weights=[p1, 1 - p1])[0] for _ in range(n)]
    initial_states = {}

    for node, state in enumerate(list_states):
        initial_states[node] = state
    
    return initial_states

# Return if there is a +1 consensus and the time to consensus
def get_consensus_type_and_time(nodes_dict):
    begin = time()

    while len(set(nodes_dict.values())) > 1:  # While not reached any consensus
        for node in nodes_dict.keys():
            # Do a cycle of state change for all nodes (based on their neighbour states)
            nodes_dict[node] = nodes_dict[random.choices(G[node])[0]] 
        if set(nodes_dict.values()) == {1}:
            consensus = 1
        else:
            consensus = 0

    end = time()
    time_to_consensus = end - begin
    return consensus, time_to_consensus

n = 1000  # Number of nodes
p = 10/n  # Probability of edge existing between two nodes
p1_results = []  # List to store p1 values

############################################################################################
################### !!! INTEFERE DIRECTLY IN SIMULATION TOTAL TIME !!! #####################
############################################################################################

n_iterations = 200  # Number of iterations to calculate probability and time of consensus for each p1 value

for p1 in [0.51, 0.55, 0.6, 0.7]:
    consensus_total_count = 0  # Variables to store number of +1 consensus and 
    consensus_total_time = 0  # total time of all iterations (the mean will be taken in the end)

    for _ in tqdm(range(n_iterations)):  # Iterate many times to calculate probability and time of consensus for each p1 value
        G = generate_graph(n, p)
        giant_component = find_giant_component(G)
        initial_states = get_initial_states(n, p1)
        # Filter only nodes present in the giant component
        initial_states = {key: initial_states[key] for key in giant_component}  
        consensus, time_to_consensus = get_consensus_type_and_time(initial_states)
        consensus_total_count += consensus
        consensus_total_time += time_to_consensus

    consensus_prob = 100 * consensus_total_count / n_iterations
    consensus_mean_time = consensus_total_time / n_iterations
    p1_results.append((p1, consensus_prob, consensus_mean_time))

    print(f"For p1 value of {p1}, the +1 consensus probability is {consensus_prob:.1f}% and the consensus mean time is {consensus_mean_time:.2f} seconds")