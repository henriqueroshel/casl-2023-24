import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

step=1e-3
us = np.arange(0,1+step,step)

def lifetime(parent, u, p, alpha):
    if u < (1-p):
        return u*parent / (1-p)
    return parent*(1+alpha*(u+p-1)/p)

if __name__=='__main__':
    parent = 10
    p = 0.5
    alpha = .3

    life = np.zeros(len(us))

    for i,u in enumerate(us):
        life[i] = lifetime(parent, u, p, alpha)

    plt.plot(us, life, markersize=1, label=f'{alpha:.2f}')
    
    # plt.yticks([0,parent,parent*(1+alpha)], ['0','$LF(d(k))$', '$LF(d(k))\\cdot(1+\\alpha)$'], rotation=45)
    plt.yticks([0,parent,parent*(1+alpha)],['','',''])
    # plt.xticks([0,1-p,1], ['0', '1-p', '1'])
    plt.xticks([0,1-p,1], ['', '', ''])
    plt.ylim([0,parent*(1+alpha)])
    plt.xlim([0,1])
    # plt.xlabel('Cumulative probability')
    # plt.ylabel('Lifespan')
    
    plt.grid()
    plt.show()

def n_simulations(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, reprod_rate, reprod_maturity, 
                    prob_improvement, alpha_improvement, minimum_accuracy=0.95, confidence_level=0.98):
    acc_popsize = 0
    # first simulation
    n_simulations = 1
    sim_species = natural_selection(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, 
                                    reprod_rate, reprod_maturity, prob_improvement, alpha_improvement)
    population_sizes = [ sim_species.average_population_size() ]

    while min(acc_popsize, acc_extinction) < minimum_accuracy:
        sim_species = natural_selection(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, 
                                        reprod_rate, reprod_maturity, prob_improvement, alpha_improvement)
        
        n_simulations += 1
        population_sizes.append( sim_species.average_population_size() )

        # confidence interval and accuracy computation
        avg_popsize, delta_popsize = confidence_interval(population_sizes, confidence_level)
        acc_popsize = accuracy(avg_popsize, delta_popsize)

        print(f'\rPopulation size: {avg_popsize:.2f} ({acc_popsize:.3f}) - Extinct: {avg_extinct_prob:.2f} ({acc_extinction:.3f})', end='', flush=True)