# COMPUTER-AIDED SIMULATIONS LAB - LAB L4

import numpy as np
import matplotlib.pyplot as plt
from random import expovariate, choice, uniform
from collections import namedtuple
from tqdm import tqdm
from scipy import stats

# Simulation conditions
NUMBER_SERVERS = 1
WAITING_LINE_SIZE = 100

# Default stop conditions constants
MAX_ARRIVALS = 5000

# other constants
ARRIVAL, DEPARTURE = 'arrival', 'departure'

# auxiliary objects
Event = namedtuple('Event', ['time', 'type'])
Client = namedtuple('Client', ['event_type', 'arrival_time'])

# definition of the patients queue combining a queue for each code
class Queue:
    def __init__(self):
        # definition FIFO queue
        self.items = []
    def __len__(self):
        return len(self.items)
    def isEmpty(self): 
        return len(self) == 0

    def put(self, patient, clock):
        # add patient to the queue
        self.items.append(patient)
        return
    def pop(self, clock):
        # pop next patient in queue
        if self.isEmpty():
            print("Queue is empty")
            return None 
        return self.items.pop(0)

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
        next_event_index = self.items.index( next_event )
        return self.items.pop( next_event_index )

def confidence_interval(values, conf_level=0.98):
    # computes the confidence interval of a particular 
    # measure where values is a list of empirical values
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    
    if n<30: # t distribution
        ci_lower,ci_upper = stats.t.interval(conf_level, df=n-1, loc=avg, scale=std/n**.5)
    else: # normal distribution
        ci_lower, ci_upper = stats.norm.interval(conf_level, loc=avg, scale=std/n**.5)
    delta = (ci_upper-ci_lower) / 2
    return avg, delta

def accuracy(value, delta):
    # computes the accuracy of a measure given its value
    # and the semi-width (delta) of the confidence interval    
    eps = delta / value # relative error
    acc = 1 - eps # accuracy
    return max(acc, 0) # return only non-negative values

def service_time_generator(ServiceDict):
    # sample a service time for the given distribution
    if ServiceDict['distribution'] == 'deterministic':
        return ServiceDict['value']
    if ServiceDict['distribution'] == 'exponential':
        return expovariate(ServiceDict['lambda'])
    if ServiceDict['distribution'] == 'hyperexponential':
        u = uniform(0,1)
        p1 = ServiceDict['p1']
        lambda1 = ServiceDict['lambda1']
        if u < p1:
            return expovariate(lambda1)
        # compute lambda2 so that the mean is 1 (ignores variance constraint)
        lambda2 = (1-p1) / (1-p1/lambda1)
        return expovariate(lambda2)

def arrival(clock, fes, queue, LambdaArrival, ServiceDict, simulation_vars):
    # sample the time until the next arrival
    inter_arrival = expovariate(LambdaArrival)
    # schedule the next arrival
    client = Client(ARRIVAL, clock)
    fes.put(Event(clock+inter_arrival, ARRIVAL))

    # managing the event
    # create a record for the client
    client = Client(ARRIVAL, clock)

    # drop client if queue at maximum capacity 
    if simulation_vars['users'] >= WAITING_LINE_SIZE:
        simulation_vars['dropped'] += 1

    else:
        # check if a server is idle to start the service
        if simulation_vars['idle_servers'] > 0:
            # make a server busy
            simulation_vars['idle_servers'] -= 1
            # register delay equals to zero
            simulation_vars['delays'].append(0)           
            # sample the service time 
            service_time = service_time_generator(ServiceDict)
            # schedule the departure of the client
            fes.put(Event(clock + service_time, DEPARTURE))
        else:
            # if no server is idle, update state variable 
            simulation_vars['users'] += 1
            # insert client in the queue
            queue.put(client, clock)

    return simulation_vars

def departure(clock, fes, queue, ServiceDict, simulation_vars):
    
    # check if there are more clients in the line
    if simulation_vars['users'] > 0:
        # get the first client from the queue
        client = queue.pop(0)
        # update state variable
        simulation_vars['users'] -= 1
        # register client delay
        simulation_vars['delays'].append(clock - client.arrival_time)
        # sample the service time 
        service_time = service_time_generator(ServiceDict)
        # schedule when the client's departure
        fes.put(Event(clock + service_time, DEPARTURE))
    else:
        # if there are no clients in the line, the server becomes idle
        simulation_vars['idle_servers'] += 1
    
    return simulation_vars

def initial_data_removal(values, remove_fraction=0.05):
    # remove the initial data keeping (1-remove_fraction) of the values 
    mean = np.mean(values)
    n = len(values)
    remove_n = int(n * remove_fraction + .5) # closest integer index
    trimmed_values =  values[ remove_n : ]
    return trimmed_values

def batch_means(values, n_batches=10, accuracy_ref=0.9):
    batches = [ [] for _ in range( min(n_batches, len(values)) ) ]

    for i,val in enumerate(values):
        batches[ i%n_batches ].append(val)
    for bat in batches:
        avg, delta = confidence_interval(bat)
        acc = accuracy(avg, delta)
        if acc < accuracy_ref:
            return avg, delta
    return avg, delta

def simulation(LambdaArrival, ServiceDict, MaxArrivalsCondition):
    
    simulation_vars = {
        'idle_servers' : NUMBER_SERVERS, # number of idle servers in the system
        'users' : 0,                   # number of clients in the queue
        'arrivals' : 0,                  # count of arrivals of clients in the system
        'dropped' : 0,                   # clients who left due to waiting line at maximum capacity
        'delays' : [],                   # delay of each client to compute the average delay
    }
    
    # initialize FIFO queue
    queue = Queue()

    # simulation clock
    clock = 0
    # list of Event objects as (time, type) in a priority queue
    fes = FutureEventSet()
    # schedule first arrival at t=0
    fes.put(Event(clock, ARRIVAL))
    
    # main loop of simulation
    while not (queue.isEmpty() and fes.isEmpty()):
        # select next event in the FES with the lowest time
        clock, event_type = fes.pop()

        if event_type == ARRIVAL:
            if simulation_vars['arrivals'] < MaxArrivalsCondition:
                # arrivals are only allowed before the MaxArrivalsCondition
                simulation_vars['arrivals'] += 1
                simulation_vars = arrival(clock, fes, queue, LambdaArrival, ServiceDict, simulation_vars)
        elif event_type == DEPARTURE:
            simulation_vars = departure(clock, fes, queue, ServiceDict, simulation_vars)
        else:
            raise NameError(f"Event type {event_type} is not defined")

    # computation of the selected measures
    remove_transient_delays = initial_data_removal( simulation_vars['delays'] )
    avg_delay = np.mean( remove_transient_delays )
    drop_probability = simulation_vars['dropped'] / simulation_vars['arrivals']
    avg_users = MaxArrivalsCondition / clock
    
    results = {'avg_delay':avg_delay, 'drop_probability':drop_probability, 'avg_users':avg_users}
    return results

def PollaczekKhinchine_delay(LambdaArrival, ServiceDict):
    if ServiceDict['distribution'] == 'deterministic':
        mu = ServiceDict['value']
        var = 0
    elif ServiceDict['distribution'] == 'exponential':
        mu = ServiceDict['lambda']
        var = mu**-2
    elif ServiceDict['distribution'] == 'hyperexponential':
        mu = 1
        p1, p2 = ServiceDict['p1'], 1 - ServiceDict['p1']
        lambda1 = ServiceDict['lambda1']
        lambda2 = (1-p1) / (1-p1/lambda1)
        var = 1 + 2*p1*p2*(1/lambda1 - 1/lambda2)**2
    
    ro = LambdaArrival / mu # utilisation
    # average delay
    W = (ro + LambdaArrival*mu*var) / (2 * (mu - LambdaArrival)) + 1/mu
    return W

def LittlesLaw(LambdaArrival, W_delay):
    # W_delay is the theoretical avg delay given by PollaczekKhinchine formula
    # return average number of users in the system
    L = LambdaArrival * W_delay


def batchmeans_simulations(LambdaArrival, ServiceDict, MaxArrivalsCondition, min_accuracy=0.95, conf_level=0.98):
    accuracy_delay = 0

    # get the values of the first two simulations
    n_simulations = 2
    results1 = simulation(LambdaArrival, ServiceDict, MaxArrivalsCondition)
    results2 = simulation(LambdaArrival, ServiceDict, MaxArrivalsCondition)
    avg_delay_list = [ results1['avg_delay'], results2['avg_delay'] ]
    drop_prob_list = [ results1['drop_probability'], results2['drop_probability'] ]
    avg_users_list = [ results1['avg_users'], results2['avg_users'] ]

    while accuracy_delay < min_accuracy:
        results = simulation(LambdaArrival, ServiceDict, MaxArrivalsCondition)
        n_simulations += 1
        avg_delay_list.append(results['avg_delay'])
        drop_prob_list.append(results['drop_probability'])
        avg_users_list.append(results['avg_users'])

        # confidence interval and accuracy computation
        avg_delay, delta_delay = confidence_interval(avg_delay_list, conf_level)
        accuracy_delay = accuracy(avg_delay, delta_delay)
        drop_prob, delta_drop_prob = confidence_interval(drop_prob_list, conf_level)
        accuracy_dropp = accuracy(drop_prob, delta_drop_prob)

        if LambdaArrival==1.2 and results['drop_probability']>0:
            print(results['drop_probability'], end=' ')

    avg_delay_batchmeans, delta_delay_batchmeans = batch_means(avg_delay_list)
    drop_prob_batchmeans, delta_drop_prob_batchmeans = batch_means(drop_prob_list)
    avg_users = np.mean(avg_users_list)
    results = {
        'avg_delay':avg_delay_batchmeans, 
        'delta_delay':delta_delay_batchmeans, 
        'drop_prob':drop_prob_batchmeans,
        'delta_drop_prob':delta_drop_prob_batchmeans,
        'avg_users':avg_users
    }


    return results

def plot_batchmeans(lambdas_arrival, avg_delays, service_dist, PollaczekKhinchine_delays=None, savedir='.\\figures'):

    fig,ax = plt.subplots()
    np.nan_to_num(avg_delays, copy=False)
    ax.plot(lambdas_arrival, avg_delays, '.', markersize=12, label='Simulation')

    if PollaczekKhinchine_delays is not None:
        ax.plot(lambdas_arrival, PollaczekKhinchine_delays, '.', markersize=12, label='PollaczekKhinchine')
        ax.set_yscale('log')
        ax.legend()        

    ax.set_title(service_dist)
    ax.set_xlabel('Rate of arrival \u03bb')
    ax.set_ylabel('Average delay')
    ax.set_xlim([min(lambdas_arrival)-0.05, max(lambdas_arrival)+0.05])
    ax.grid(alpha=0.5)

    filename = f'.\\serv_{service_dist}.png'
    try:
        fig.savefig(f'{savedir}{filename}', format='png')
    except:
        fig.savefig(filename, format='png')
    return

def plot_MM1(lambdas_arrival, avg_delays, drop_probs, savedir='.\\figures'):

    fig,ax = plt.subplots()
    ax.plot(lambdas_arrival, avg_delays, '.', markersize=12)
    ax.grid()
    ax.set_xlabel('Rate of arrival \u03bb')
    ax.set_ylabel('Average delay')

    try:
        fig.savefig(f'{savedir}.\\MM1delay{WAITING_LINE_SIZE}.png', format='png')
    except:
        fig.savefig(f'MM1delay{WAITING_LINE_SIZE}.png', format='png')    

    fig,ax = plt.subplots()
    ax.plot(lambdas_arrival, drop_probs, '.', markersize=12)
    ax.grid()
    ax.set_xlabel('Rate of arrival \u03bb')
    ax.set_ylabel('Dropping probability')

    try:
        fig.savefig(f'{savedir}.\\MM1drop{WAITING_LINE_SIZE}.png', format='png')
    except:
        fig.savefig(f'MM1drop{WAITING_LINE_SIZE}.png', format='png')        

    return

if __name__ == '__main__':

    max_arrivals = MAX_ARRIVALS
    print(f'Maximum number of arrivals: {MAX_ARRIVALS}')

    # EXPERIMENT 1
    service_deterministic = {'distribution' : 'deterministic', 'value' : 1}
    service_exponential = {'distribution' : 'exponential', 'lambda' : 1}
    service_hyperexp = {'distribution' : 'hyperexponential', 'p1':0.35, 'lambda1':0.75}
    lambdas_arrival = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]

    fig, axs = plt.subplots(3,sharex=True)
    axs[2].set_xlabel('Rate of arrival \u03bb')
    Services = [ service_deterministic, service_exponential, service_hyperexp ]
    for ServiceDict in Services:

        avg_delays = np.zeros(len(lambdas_arrival))
        PollaczekKhinchine_delays = np.zeros(len(lambdas_arrival))
        LittlesLaw_difference = np.zeros(len(lambdas_arrival))

        for i, LambdaArrival in enumerate(tqdm(lambdas_arrival)):

            results = batchmeans_simulations(
                        LambdaArrival=LambdaArrival, 
                        ServiceDict=ServiceDict,
                        MaxArrivalsCondition=max_arrivals, 
                        min_accuracy=0.95, 
                        conf_level=0.98
                                            )
            
            avg_delays[i] = results['avg_delay']
            PollaczekKhinchine_delays[i] = PollaczekKhinchine_delay(LambdaArrival, ServiceDict)

            # Little's Law
            W = PollaczekKhinchine_delays[i]
            LittlesLaw = W * LambdaArrival
            LittlesLaw_difference[i] = LittlesLaw - results['avg_users']

        plot_batchmeans(lambdas_arrival, avg_delays, ServiceDict['distribution'],)
                        # PollaczekKhinchine_delays=PollaczekKhinchine_delays)
        
        axs[ Services.index(ServiceDict) ].set_yscale('log')
        axs[ Services.index(ServiceDict) ].plot(lambdas_arrival, LittlesLaw_difference)
        axs[ Services.index(ServiceDict) ].grid(alpha=.5)
        axs[ Services.index(ServiceDict) ].set_ylabel(ServiceDict['distribution'])
        
        try:
            fig.savefig(f'.\\figures\\littleslaw.png', format='png')
        except:
            fig.savefig(f'littleslaw.png', format='png')

    # EXPERIMENT 2
    service_exponential = {'distribution' : 'exponential', 'lambda' : 1}
    lambdas_arrival = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1, 1.1, 1.2]

    avg_delays = np.zeros(len(lambdas_arrival))
    drop_probs = np.zeros(len(lambdas_arrival))

    for i, LambdaArrival in enumerate(tqdm(lambdas_arrival)):
        results = batchmeans_simulations(
            LambdaArrival=LambdaArrival, 
            ServiceDict=service_exponential,
            MaxArrivalsCondition=max_arrivals, 
            min_accuracy=0.95, 
            conf_level=0.98
                )
        
        avg_delays[i] = results['avg_delay']
        drop_probs[i] = results['drop_prob']    

    print(lambdas_arrival, avg_delays, drop_probs, sep='\n')
    plot_MM1(lambdas_arrival, avg_delays, drop_probs)