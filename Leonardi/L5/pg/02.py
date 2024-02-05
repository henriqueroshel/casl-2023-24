
import heapq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, probplot

class GraphSimulator:
    def __init__(self, n, p=None, is_grid=False):
        self.n = n
        self.p = p
        self.is_grid = is_grid
        self.graph = None
        self.fes = []  # Future Event Set, implemented as a heap

    def generate_graph(self):
        if self.is_grid:
            # Generating a regular grid
            side = int(np.sqrt(self.n))
            self.graph = nx.grid_2d_graph(side, side)
            self.graph = nx.convert_node_labels_to_integers(self.graph)
        else:
            # Generating a G(n, p) graph
            self.graph = nx.erdos_renyi_graph(self.n, self.p)

    def schedule_event(self, event_time, event_data):
        heapq.heappush(self.fes, (event_time, event_data))

    def process_next_event(self):
        if self.fes:
            return heapq.heappop(self.fes)
        return None

    # Additional methods for running simulation and handling events can be added here

    def analyze_degree_distribution(self):
        degrees = [d for _, d in self.graph.degree()]
        empirical_dist = np.bincount(degrees, minlength=self.n) / self.n
        expected_dist = np.random.binomial(self.n - 1, self.p, self.n) / self.n

        # Chi-squared test
        chi2_stat, p_val = chisquare(empirical_dist, expected_dist)

        # Q-Q plot
        plt.figure()
        probplot(degrees, dist="binom", sparams=(self.n - 1, self.p), plot=plt)
        plt.title('Q-Q Plot of Degree Distribution')
        plt.show()

        return chi2_stat, p_val

# Example Usage
n=10_000
p=10/n
simulator = GraphSimulator(n,p)
simulator.generate_graph()
chi2_stat, p_val = simulator.analyze_degree_distribution()
print(f"Chi-squared Statistic: {chi2_stat}, P-value: {p_val}")
