import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple
from tqdm import tqdm   # progress bar
from itertools import count

# constants for event types
BIRTH, DEATH = "BIRTH", "DEATH"
# event object as a tuple (time, type of event, individual)
Event = namedtuple('Event', ['time', 'event', 'individual'])
class FutureEventSet:
    # implementation of the Future Event Set as a priority queue
    def __init__(self):
        self.items = []
    def __len__(self):
        return len(self.items)
    def isEmpty(self):
        return len(self) == 0
        
    def put(self, event):
        self.items.append(event)
    def pop(self):
        # pop next event (lowest time) if there is events on the FES
        if self.isEmpty():
            print("FutureEventSet is empty")
            return   
        next_event = min(self.items, key=lambda ev:ev.time)
        self.items.remove( next_event )
        return next_event

def confidence_interval(values, conf_level=0.98):
    # computes the confidence interval of a particular 
    # measure where values is a list of empirical values
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    
    if n<30: # t distribution
        ci_lower,ci_upper = stats.t.interval(conf_level, df=n-1, loc=avg, scale=std/n**.5)
    else:# normal distribution
        ci_lower,ci_upper = stats.norm.interval(conf_level, loc=avg, scale=std/n**.5)
    delta = (ci_upper-ci_lower) / 2
    return avg, delta
def accuracy(value, delta):
    # computes the accuracy of a measure given its value
    # and the semi-width (delta) of the confidence interval    
    eps = delta / value # relative error
    acc = 1 - eps # accuracy
    return max(acc, 0) # return only non-negative values

class Individual:
    # model for the individual of the considered species
    individual_id = count()
    def __init__(self, birth_time, species, parent=None, lifetime=None):
        self._id = next(Individual.individual_id)
        self.parent = parent
        self.generation = parent.generation+1 if parent else 0
        self.offspring = 0 # number of children
        self.birth_time = birth_time
        self.species = species
        self.lifetime = lifetime

    def __hash__(self):
        return self._id
    def __repr__(self):
        return f'Individual({self._id})'
    def __str__(self):
        s = f'Individual({self._id}, birthtime={self.birth_time:.2f}, lifetime={self.lifetime:.2f}, gen={self.generation}, offspring={self.offspring}'
        s+= f', parent={repr(self.parent)}' if self.parent else ''
        return s+')'

    def generate_LF(self, prob_improvement, alpha_improvement):
        # inverse transform method for the given distribution
        lf_parent = self.parent.lifetime
        p, alpha = prob_improvement, alpha_improvement
        u = np.random.uniform(0,1)
        
        # if resources are limited, chance of survival decreases
        pen = self.species.resources_limitation_pen()

        if u < (1-p):
            life = lf_parent * (u/(1-p))
        else:
            life = lf_parent * (1 + alpha*(u+p-1)/p)
        life = life * (1-pen)
        if pen==1:
            print(life, end='\n')
        return life
    
    def birth(self):
        if self.generation > 0: # generation 0 is instantiated already with its life time
            self.lifetime = self.generate_LF(self.species.prob_improvement, self.species.alpha_improvement)
            if self.lifetime>0:
                # increment the number of children of the parent individual
                self.parent.offspring += 1
        self.death_time = self.birth_time + self.lifetime

    def next_child(self, clock):
        # generate next child of individual as a poisson process,
        # i.e. time between birth is exponentially distributed
        child_birth_time = clock + np.random.exponential(1 / self.species.reprod_rate)
        reprod_maturity = self.species.reprod_maturity
        if clock-self.birth_time<reprod_maturity and self.generation!=0:
            # wait reproduction maturity for the first child on succeeding generations
            # assume maturity already reached for generation 0
            child_birth_time += reprod_maturity
        child = Individual(child_birth_time, self.species, parent=self)
        if child.birth_time < self.death_time:
            return child
            # ignore if child is to be born after individual's death

class Species:
    # class for handling a species and storing its characteristics
    def __init__(self, reprod_rate, reprod_maturity, prob_improvement, alpha_improvement, population_size_threshold):
        self.population_size = 0
        self.population_log = dict() # time:population size
        self.reprod_rate = reprod_rate
        self.reprod_maturity = reprod_maturity
        self.prob_improvement = prob_improvement
        self.alpha_improvement = alpha_improvement
        self.population_size_threshold = population_size_threshold

        self.is_extinct = False

    def birth(self, clock, individual, fes):
        # birth event for individual
        individual.birth()
        # updates log of population
        self.population_size += 1
        self.population_log[clock] = self.population_size 

        # schedule next child of individual's parent
        if individual.parent:
            brother = individual.parent.next_child(clock)
            if brother: # birth before parent's death
                fes.put( Event(brother.birth_time, BIRTH, brother) )

        # schedule first child of individual (after reproduction maturity)
        child = individual.next_child(clock)
        if child: # birth before individual death
            fes.put( Event(child.birth_time, BIRTH, child) )
        # schedule individual's death
        fes.put( Event(individual.death_time, DEATH, individual) )    

    def death(self, clock, individual):
        # death event for individual
        self.population_size -= 1
        self.population_log[clock] = self.population_size
        if self.population_size == 0:
            self.is_extinct = True
    
    def resources_limitation_pen(self):
        # penalize individuals lifespan if the population grows over the population size threshold
        penalty = (self.population_size-self.population_size_threshold) / self.population_size_threshold
        penalty = max(0,penalty) # population size lower than threshold
        penalty = min(penalty,1) # population size greater than 2*threshold
        return penalty

    def average_population_size(self):
        # computes the average population size given the log of the simulation
        n = len(self.population_log) # number of entrances in the log
        time_log = np.fromiter( self.population_log.keys(), dtype='f' )
        size_log = np.fromiter( self.population_log.values(), dtype='i' )
        max_time = np.max(time_log)
        max_size = np.max(size_log)
        # stores the product of time by population size (size equal to the list index)
        time_by_size = np.zeros(max_size+1)
        for i in range(n-1):
            size = size_log[i]
            time_by_size[size] += (time_log[i+1]-time_log[i]) * size
        # compute as a weighted average 
        avg_pop_size = np.sum(time_by_size) / max_time
        return avg_pop_size

    def plot_population(self, savedir='.\\figures'):
        # plot population size evolution
        fig,ax = plt.subplots()
        time_log = np.fromiter( self.population_log.keys(), dtype='f' )
        size_log = np.fromiter( self.population_log.values(), dtype='i' )

        avg_pop_size = self.average_population_size()
        ax.step(time_log, size_log, where='post', label='population size evolution', c='C2')
        ax.axhline(avg_pop_size, ls='-.', label='average population size', lw=.9, c='C3')
        ax.text(max(time_log)*.01, avg_pop_size*1.01, f'{avg_pop_size:.2f}', fontsize=9, c='C3')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Population size')
        ax.set_xlim([0,max(time_log)])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc='best')
        ax.grid()
        plt.show()
        
        try:
            fig.savefig(f'{savedir}\\natselection.png', format='png')
        except:
            fig.savefig('.\\natselection.png', format='png')

def natural_selection(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, 
                        reprod_rate, reprod_maturity, prob_improvement, alpha_improvement, verbose=True):
    species = Species( reprod_rate, reprod_maturity, prob_improvement, alpha_improvement, population_size_threshold )
    # we consider the initial distribution of the remaining lifetime of the population to be normally distributed
    initial_population_LF = np.random.normal(initial_lifetime_mean, scale=1, size=P)

    fes = FutureEventSet()
    for i in range(P):
        # instantiate each individual of the initial population
        individual = Individual(0, species=species, lifetime=initial_population_LF[i])
        fes.put( Event(0, BIRTH, individual) )
    
    clock=0
    while clock<max_simulation_time and not fes.isEmpty():
        (clock, event, individual) = fes.pop()
        if verbose:
            print(f'\rClock: {clock:.3f} - Population size: {species.population_size}', end=' ', flush=True)

        if event==BIRTH:
            species.birth(clock, individual, fes)
        elif event==DEATH:
            species.death(clock, individual)
        else:
            raise ValueError(f"Event type {event} is not defined.")
    if verbose:
        print(f'\rClock: {clock:.3f} - Population size: {species.population_size} ', flush=True)
    return species

def n_simulations(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, reprod_rate, reprod_maturity, 
                  prob_improvement, alpha_improvement, N_sims, conf_level=0.98, verbose=True):
    # perform N_sims simulations and computes the output measures
    avg_population_size_list = np.zeros(N_sims)
    final_population_size_list = np.zeros(N_sims)
    extinct_list = np.zeros(N_sims)

    if verbose:
        print(f'INPUT PARAMETERS \n- Max simulation time: {max_simulation_time}\n- Initial population size: {P}')
        print(f'- Initial population average lifetime: {initial_lifetime_mean}\n- Reproduction rate: {reprod_rate}')
        print(f'- Reproduction maturity: {reprod_maturity}\n- Probability of improvement: {prob_improvement:.0%}')
        print(f'- Improvement factor (\u03b1): {alpha_improvement}')

    for i in tqdm(range(N_sims)):
        species_sim = natural_selection(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, 
                                    reprod_rate, reprod_maturity, prob_improvement, alpha_improvement, verbose=False)
        avg_population_size_list[i] =species_sim.average_population_size()
        final_population_size_list[i] =  species_sim.population_size 
        extinct_list[i] = int(species_sim.is_extinct)


    # confidence interval and accuracy computation
    avg_pop_size, delta_avg_pop_size = confidence_interval(avg_population_size_list, conf_level)
    acc_avg_population_size = accuracy(avg_pop_size, delta_avg_pop_size)
    final_pop_size, delta_final_pop_size = confidence_interval(final_population_size_list, conf_level)
    acc_final_population_size = accuracy(final_pop_size, delta_final_pop_size)
    extinction_prob, delta_extinct = confidence_interval(extinct_list, conf_level)
    acc_extinction = accuracy(extinction_prob, delta_extinct)

    if verbose:
        print(f'\n OUTPUT MEASURES\n- Average population size: {avg_pop_size:.2f} \u00b1 {delta_avg_pop_size:.2f} (accuracy: {acc_avg_population_size:.2%})')
        print(f'- Final population size: {final_pop_size:.2f} \u00b1 {delta_final_pop_size:.2f} (accuracy: {acc_final_population_size:.2%})')
        print(f'- Extinction probability: {extinction_prob:.3f} \u00b1 {delta_extinct:.3f} (accuracy: {acc_extinction:.2%})')

if __name__=='__main__':
    np.random.seed(40028922)
    
    # INPUT PARAMETERS
    reprod_rate = 0.5
    reprod_maturity = 2.5
    prob_improvement = 0.15
    alpha_improvement = 0.25
    P = 10
    population_size_threshold = 1000
    initial_lifetime_mean = 7.5

    # perform one single simulation (longer simulation period)
    max_simulation_time = 150
    species = natural_selection(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, reprod_rate, 
                            reprod_maturity, prob_improvement, alpha_improvement)
    avg_pop_size = species.average_population_size()
    print(f'Average population size: {avg_pop_size:.2f}')  
    print(f'Maximum population size: {max(species.population_log.values()):.2f}')  
    species.plot_population()

    # perform N_sims simulations (reduced simulation period)
    max_simulation_time = 50
    N_sims=2000
    n_simulations(P, max_simulation_time, population_size_threshold, initial_lifetime_mean, reprod_rate, reprod_maturity, 
                  prob_improvement, alpha_improvement, N_sims=N_sims)