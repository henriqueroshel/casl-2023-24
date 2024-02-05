# run the file -> python lab6.py
# part 1 was using about 4 hours to run
# part 2 was using about 3 hours to run

import numpy as np
import scipy.sparse
import networkx as nx
from scipy.stats import t
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


def ci_t(data, confidence_level=0.90):  #confidence interval
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate t-score for the given confidence level
    degrees_of_freedom = len(data) - 1
    t_score = t.ppf(1 - (1 - confidence_level) / 2, df=degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_score * (std_dev / np.sqrt(len(data)))
    
    # Calculate accuracy
    accuracy = 1 - margin_of_error / mean

    return accuracy, mean


def initialize_graph(n, p):
    # Creates an n×n empty sparse matrix to represent the adjacency matrix of the graph.
    # lil_matrix -> for dynamic modification during construction.
    mat = scipy.sparse.lil_matrix((n, n))

    # determining the number of edges in the graph, according to the Poisson distribution
    num_edges = np.random.poisson(p * (n)**2)

    # Loops for the determined number of edges.
    for k in range(num_edges):
        # Randomly selects indices of two nodes.
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

        # If the selected nodes are already connected or the same node, reselect nodes.
        while i == j or mat[i, j] != 0:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)

        # Connects the chosen nodes by setting the corresponding position in the matrix to 1.
        mat[i, j] = 1

    # Converts the lil_matrix to a NetworkX graph for further analysis
    G = nx.Graph(mat)

    return G


def initialize_states(graph, p_1):    
    state = {}
    for node in graph:
        # Initialize state based on p_1
        state[node] = np.random.choice([-1, 1], p=[1 - p_1, p_1])

    return state


def find_neighbors(G):
    neighbors = {}

    # Iterate over each node in the graph
    for node in G.nodes():
        
        # Get the list of neighbors for the current node
        neighbor = list(G.neighbors(node))
        
        # Assign the list of neighbors to the current node in the dictionary
        neighbors[node] = neighbor

    return neighbors


def find_neighbors_v2(graph, k, size):
    # empty dictionary to store neighbors for each node
    neighb = {}
    
    # Iterate over each node in the graph
    for node in graph:
        # empty list to store neighbors of the current node
        neighbors = []
        
        # Iterate over possible neighbors in the k-dimensional space
        for delta in itertools.product([-1, 0, 1], repeat=k):

            # Calculate the coordinates of the potential neighbor
            neighbor = tuple(p + d for p, d in zip(node, delta))
            
            # Check if the neighbor is within the valid range and is different from the current node
            if all(0 <= coord < size for coord in neighbor) and neighbor != node and sum(abs(d) for d in delta) == 1:
                
                # Add the valid neighbor to the list
                neighbors.append(neighbor)
        
        # Assign the list of neighbors to the current node in the dictionary
        neighb[node] = neighbors
    
    return neighb


def k_grid(k):
    # Check if the dimension is 2
    if k == 2:
        # Set the size for a 2-D grid, means 31*31 = 961 nodes
        s = 31
        # Generate the grid using itertools.product
        return list(itertools.product(range(s), repeat=k)), s
    else:
        # 3-D  10*10*10 = 1000 nodes
        s = 10
        return list(itertools.product(range(s), repeat=k)), s


def voter_model_v1(p_1, n, sim_time):

    consensus_1 = 0
    time_list = []
    p = 10 / n

    for _ in range(sim_time):                               # run the model in sim_time times to get the prob of reaching a +1 consensus 
        time = 0                                            # record the time to reach consensus
        consensus = False
        G = initialize_graph(n, p)

        if nx.is_connected(G):
            state_dict = initialize_states(G, p_1)          # record the state of all nodes
            neighbor_dict = find_neighbors(G)               # record the neighbors of all nodes
            # This graph is connected
        else:
            # This graph is not connected. we only consider the giant component
            components = list(nx.connected_components(G))
            # find the giant component
            largest_component = max(components, key=len)
            # create the subgraph
            G = G.subgraph(largest_component)
            state_dict = initialize_states(G, p_1)          # record the state of all nodes
            neighbor_dict = find_neighbors(G)               # record the neighbors of all nodes

        while not consensus:                                # keep interation all nodes til consensus

            for node in list(state_dict.keys()):    

                #  random choose 1 neighbor
                random_neighbor = random.choice(neighbor_dict[node])
                # update the state with copy of neighbor's
                state_dict[node] = state_dict[random_neighbor]

                # Check if all nodes have the same state
                consen_state = set(state_dict.values())
                if len(consen_state) == 1:
                    # print(f'Consensus got! with state {consen_state}')
                    consensus = True
                    if list(state_dict.values())[0] == 1:           # if consensus with +1
                        consensus_1 += 1
                    break
                time += np.random.exponential(1)

        time_list.append(time)

    return consensus_1/sim_time, np.mean(time_list)


def voter_model_v2(k, p_1, sim_time):

    consensus_1 = 0
    time_list = []

    for _ in range(sim_time):                                # run the model in 10 times to get the prob of reaching a +1 consensus 
        time = 0                                            # record the time to reach consensus
        consensus = False
        G, s = k_grid(k)
        state_dict = initialize_states(G, p_1)
        neighbor_dict = find_neighbors_v2(G, k, size=s)

        while not consensus:                                # keep interation all nodes til consensus

            for node in list(state_dict.keys()):    

                #  random choose 1 neighbor
                random_neighbor = random.choice(neighbor_dict[node])
                # update the state with copy of neighbor's
                state_dict[node] = state_dict[random_neighbor]

                # Check if all nodes have the same state
                consen_state = set(state_dict.values())
                if len(consen_state) == 1:
                    # print(f'Consensus got! with state {consen_state}')
                    consensus = True
                    if list(state_dict.values())[0] == 1:           # if consensus with +1
                        consensus_1 += 1
                    break
                time += np.random.exponential(1)

        time_list.append(time)

    return consensus_1/sim_time, np.mean(time_list)


def simulation(max_sim):
    print('Part 1')
    p_1_values = [0.51, 0.55, 0.6, 0.7]
    prob_tot_v1 = []
    time_tot_v1 = []
    # p_1_values = [0.6]
    for p_1 in p_1_values:
        prob_list = []
        time_list = []
        print(f'The prob of state 1 is {p_1}')
        for i in tqdm(range(max_sim)):
                
            prob, time = voter_model_v1(p_1, 10**3, 20)        # part 1 of the lab
            time_list.append(time)
            prob_list.append(prob)

            if i >= 2:                              # if there are more than 2 sims
                acc, prob_mean = ci_t(prob_list, confidence_level=0.9)     # compute the accuracy
                if acc > 0.9:
                    print(f'the accuracy: {acc}')
                    prob_tot_v1.append(prob_mean)
                    time_tot_v1.append(np.mean(time_list))
                    break
    print('Part 2')
    k_values = [2, 3]
    prob_tot_v2 = []
    time_tot_v2 = []
    p_1 = 0.51
    prob_tot_v2.append(prob_tot_v1[0])
    time_tot_v2.append(time_tot_v1[0])
    for k in k_values:
        prob_list = []
        time_list = []
        print(f'The k is {k}')
        for i in tqdm(range(max_sim)):
                
            prob, time = voter_model_v2(k, p_1, 20)       # part 2 of the lab
            time_list.append(time)
            prob_list.append(prob)

            if i >= 2:                              # if there are more than 2 sims
                acc, prob_mean = ci_t(prob_list, confidence_level=0.9)     # compute the accurancy
                if acc > 0.9:
                    print(f'the accuancy: {acc}')
                    prob_tot_v2.append(prob_mean)
                    time_tot_v2.append(np.mean(time_list))
                    break
    return prob_tot_v1, time_tot_v1, prob_tot_v2, time_tot_v2



prob_1, time_1, prob_2, time_2 = simulation(30)
p_1_values = [0.51, 0.55, 0.6, 0.7]
print(prob_1, time_1, prob_2, time_2)

plt.figure()
plt.plot(p_1_values, prob_1)
plt.scatter(p_1_values, prob_1)
plt.xlabel('Prob of state equal to 1')
plt.ylabel('Prob of consensus as 1')
plt.show()
plt.close()

plt.figure()
plt.plot(p_1_values, time_1)
plt.scatter(p_1_values, time_1)
plt.xlabel('Prob of state equal to 1')
plt.ylabel('Time to reach consensus')
plt.show()
plt.close()

plt.figure()
plt.plot([1,2,3], prob_2)
plt.scatter([1,2,3], prob_2)
plt.xlabel('k')
plt.ylabel('Prob of consensus as 1')
plt.show()
plt.close()

plt.figure()
plt.plot([1,2,3], time_2)
plt.scatter([1,2,3], time_2)
plt.xlabel('k')
plt.ylabel('Time to reach consensus')
plt.show()
plt.close()