from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as stats
import statsmodels.api as sm

@dataclass(order=True)
class Event:
    time: float
    node: int = field(compare=False)
    action: Callable = field(compare=False)

class Graph:

    def __init__(self, n_vertices: int, p: float, seed: int = 42) -> None:
        self.n = n_vertices
        self.p = p 
        self.edges = defaultdict(list)
        self.gen = np.random.default_rng(seed)
        self.seed = seed

    def add_edge(self, v1: int, v2: int, directed: bool = False) -> None:
       self.edges[v1].append(v2) 
       if not directed:
        self.edges[v2].append(v1)

    def __str__(self) -> str:
        return f"G(n={self.n}, p={self.p}), seed={self.seed} and edges={self.edges}"

class SimulatedGraph(Graph):

    def __init__(self, n_vertices: int, p: float, lambd: float, init_status: list, seed: int = 42) -> None:
        assert len(init_status) == n_vertices, "Lenght of initial status doesn't match the number of vertices"
        super().__init__(n_vertices, p, seed)
        self.nodes_status = deepcopy(init_status)
        self.time = 0
        self.fes = PriorityQueue[Event]()
        self.lambd = lambd
        for i in range(self.n):
            self._schedule_wake_up(i)


    @staticmethod
    def generate(n: int, p: float, lambd: float, init_status: list, seed: int = 42):
        g = SimulatedGraph(n, p, lambd, init_status, seed)
        v = 1
        w = -1
        log_q = np.log(1-p)
        while v < n:
            r = g.gen.uniform(0,1)
            w += 1 + int(np.ceil(np.log(1-r) / log_q))
            while w >= v and v < n:
                w -= v
                v += 1
            if v < n:
                g.add_edge(v, w)
        return g

    def _schedule_wake_up(self, node_id: int) -> None:
        inter_time = self.gen.exponential(1 / self.lambd)
        event_time = self.time + inter_time
        self.fes.put(Event(event_time, node_id, self.wake_up))

    def wake_up(self, node_id: int) -> None:
        self._schedule_wake_up(node_id)
    
    def run(self, end_time: int) -> None:
        while self.time < end_time:
            event = self.fes.get()
            self.time = event.time
            event.action(event.node)

def plot_distributions(graph: Graph):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    nbins = 15
    degrees = [len(vertices) for vertices in graph.edges.values()]
    x = np.arange(stats.binom.ppf(0.01, graph.n, graph.p), stats.binom.ppf(0.99, graph.n, graph.p))
    
    analytical_pdf = stats.binom.pmf(x, graph.n, graph.p)
    analytical_cdf = stats.binom.cdf(x, graph.n, graph.p)
    
    axs[0].hist(degrees, density=True, bins=nbins, histtype='step', label="Empirical cdf", cumulative=True)
    axs[0].plot(x, analytical_cdf, label='Analytical cdf', color='red', alpha=0.5, linestyle='--')
    axs[0].legend()
    axs[1].plot(x, analytical_pdf, label='Analytical pmf', color='red', alpha=0.5, linestyle='--')
    axs[1].hist(degrees, density=True, bins=nbins, histtype='step', label="Empirical pmf")
    axs[1].legend()
    plt.show()

def qq_plot(graph: Graph):
    #fig, ax = plt.subplots()
    degrees = [len(vertices) for vertices in graph.edges.values()]
    #dist=stats.binom, distargs=(graph.n, graph.p)
    fig = sm.qqplot(np.array(degrees), line="45")
    plt.show()
    #stats.probplot(degrees, dist="binom", plot=plt)    

if __name__ == "__main__":
    n = 50_000
    p = 10/n
    lambd = 0.03
    sim_time = 1000
    init_status = [0 for i in range(n)] 
    print("Starting graph generation")
    start_time = time.time()
    g = SimulatedGraph.generate(n, p, lambd, init_status)
    print(f"--- Generation finished {time.time() - start_time} ---")
    print("Starting simulation")
    start_time = time.time()
    g.run(sim_time)
    print(f"--- Simulation finished {time.time() - start_time} ---")
    plot_distributions(g)
    qq_plot(g)
