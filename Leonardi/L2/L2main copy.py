# COMPUTER-AIDED SIMULATIONS LAB
# LAB L2 - Hospital Emergency Room (Queuing System) 

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple

# constants
ARRIVAL, DEPARTURE = 'arrival', 'departure'
RED, YELLOW, GREEN = 'RED', 'YELLOW', 'GREEN'

# Event class defined as a tuple
Event = namedtuple('Event', ['time', 'type', 'patient'])

# Randomize the code color of patients
def patient_code_generator():
    # Considering the ratio between patients code 
    # (RED:YELLOW:GREEN)=(1:2:3), generates a uniform 
    # distributer number between zero and one
    # return the code for the patient.
    rd = rand.uniform(0,6)
    # equivalently we could call rand.choices([RED,YELLOW,GREEN], weights=[1,2,3])
    if rd<1/6:
        return RED
    if rd<1/2:
        return YELLOW
    return GREEN

class Patient:
    def __init__(self, LambdaService, ArrivalTime):
        self.arrival_time = ArrivalTime
        # generate urgency code
        self.code = patient_code_generator()                    
        # sample the service time based on the urgency code
        self.service_duration = rand.expovariate(LambdaService[self.code])
        # variables to track patient treatment
        self.delay = 0 # time spent on the queue
        self.departure_time = None
        self.service_start_time = None # last time the service started
        self.remaining_time = None
        # variables for interrupted patients
        self.interruptions = 0
        self.interruption_time = None # last time the service was interrupted
    
    def start_service(self, current_time):
        self.service_start_time = current_time
        # register patient's delay and update its departure
        if self.interruptions == 0:
            self.delay += current_time-self.arrival_time
            self.departure_time = current_time + self.service_duration
        else: 
            self.delay += current_time-self.interruption_time
            self.departure_time = current_time + self.remaining_time

class Servers:
    # class to handle the state of the servers (check if there 
    # are idle servers, or when a green/yellow patient must 
    # be interrupted to give space to a red one)
    def __init__(self, K):
        self.K = K # number of servers
        # store a list with patients of each code
        self.patients = { RED:[], YELLOW:[], GREEN:[] }
    def __len__(self):
        return sum(len(self.patients[code]) for code in self.patients)
 
    def len_by_code(self, code):
        # returns length of queue of respective urgency code
        return len(self.patients[code])
    def at_least_one_idle(self):
        # True if there is at least one idle server
        return len(self) < self.K
    def at_least_one_non_priority(self):
        # True if there is at least one patient with code 
        # different than red receiving treatment (ignore idle servers)
        return len(self.patients[RED]) < len(self)

    # add a patient to the servers (start of service)
    def put(self, patient):
        if self.at_least_one_idle():
            code = patient.code
            self.patients[code].append(patient)
        else:
            print("No idle servers")
    
    # remove a patient from the servers (end of treatment)
    def remove(self, patient):
        code = patient.code
        self.patients[code].remove(patient)

    # pop and return the last non-priority patient to begin  
    # the treatment for the treatment of a red code patient. 
    # A yellow code patient will only be interrupted if
    # there are no green code patients on any server 
    def pop(self):
        if self.at_least_one_non_priority():
            if self.len_by_code(GREEN) > 0:
                return self.patients[ GREEN ].pop(-1)
            return self.patients[ YELLOW ].pop(-1)
        else:
            print("Only very urgent treatments are on course")

# class for storing the length of the queue along time
Log = namedtuple('Log', ['time', 'total', RED, YELLOW, GREEN])
# definition of the patients queue combining a queue for each code
class PatientsQueue:
    def __init__(self):
        # define one queue for each code (similar to FIFO, except that 
        # interrupted patients are inserted on the front of the queue)
        self.items = { RED:[], YELLOW:[], GREEN:[] }
        # tracks the length of each queue
        self.log = Log([0],[0],[0],[0],[0])
    def __len__(self):
        return sum( len(q) for q in self.items.values() )

    def isEmpty(self): 
        return len(self) == 0
    def len_by_code(self, code):
        # returns length of queue of respective urgency code
        return len(self.items[code])
    def update_log(self, clock):
        # register the queues state on the log
        r,y,g = self.len_by_code(RED), self.len_by_code(YELLOW), self.len_by_code(GREEN) 
        self.log.time.append(clock)
        self.log.total.append(r+y+g)
        self.log.RED.append(r)
        self.log.YELLOW.append(y)
        self.log.GREEN.append(g)

    def put(self, patient, clock):
        # add patient to the corresponding queue;
        code = patient.code
        # if patient just arrived (not interrupted) or 
        # its queue is empty he goes to the back of the queue
        if patient.interruptions==0 or self.len_by_code(code)==0:
            self.items[code].append(patient)
            self.update_log(clock)
            return
        # if the patient has been interrupted, he enters on the front of the line
        # (behind the patients who also have been interrupted)
        for i, pat_in_q in enumerate(self.items[code]):
            if pat_in_q.interruptions == 0:
                self.items[code].insert(i, patient)
                self.update_log(clock)
                return
                    
    def pop(self, clock):
        # pop next patient in queue checking the queues in order of priority 
        # check RED, YELLOW and then GREEN queue
        if self.isEmpty():
            print("Queue is empty")
            return None 
        if self.len_by_code(RED) > 0:
            patient = self.items[RED].pop(0)
            self.update_log(clock)
            return patient
        if self.len_by_code(YELLOW) > 0:
            patient = self.items[YELLOW].pop(0)
            self.update_log(clock)
            return patient
        if self.len_by_code(GREEN) > 0:
            patient = self.items[GREEN].pop(0)
            self.update_log(clock)
            return patient

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
    
    def filter_fes(self, event_type=None, event_time=None, patient=None):
        # filter the set of future events based on given event conditions
        if not any([event_type, event_time, patient]):
            print("At least one condition is necessary to filter")
            return    
        filtered_fes = self.items.copy()
        if event_time:
            filtered_fes = filter(lambda ev:ev.time==event_time, filtered_fes)
        if event_type:
            filtered_fes = filter(lambda ev:ev.type==event_type, filtered_fes)
        if patient:
            filtered_fes = filter(lambda ev:ev.patient==patient, filtered_fes)
        return list(filtered_fes)

    def remove(self, event_type=None, event_time=None, patient=None):
        # remove events that match given conditions
        # filter the events to be removed
        remove_from_fes = self.filter_fes(event_type, event_time, patient)
        new_items = [ev for ev in self.items if ev not in remove_from_fes]
        self.items = new_items

# ROUTINES TO RUN THE SIMULATION
def simulation(LambdaArrival, LambdaService, NumberServers, MaxTimeCondition):
    '''
    Simulates the Hospital Emergency Room queuing system
    Arguments:
        LambdaArrival (float) : rate of arrival of patients 
            (not considering urgency code)
        LambdaService (dict) : rate of treatment of patients
            keys RED, YELLOW and GREEN individualize the treatment
            rate based on the urgency code
        MaxTimeCondition (float) : 
            last possible instant of simulation for the arrival of patient;
            after that, the remaining patients on the queue receive treatment
            until the queue is empty
    Return: 
        departure_patients (list) : list of patients treated during the simulation
        queue_log (dict) : log of queue length during the simulation registered 
            within the PatientsQueue object.
    '''
    # list of patients whose treatment finished
    departure_patients = []

    # simulation clock
    clock = 0
    # instances of FutureEventSet, PatientsQueue and Servers
    fes = FutureEventSet()
    queue = PatientsQueue()
    servers = Servers(K=NumberServers)
    
    # schedule arrival of first patient at t=0
    first_patient = Patient(LambdaService, clock)
    fes.put(Event(clock, ARRIVAL, first_patient))

    # main loop of simulation
    while not (queue.isEmpty() and fes.isEmpty()):
        # select next event in the FES with the lowest time
        clock, event_type, patient = fes.pop()
        if event_type == ARRIVAL: 
            if clock < MaxTimeCondition:
                # arrivals are only allowed before the MaxTimeCondition
                arrival(clock, patient, fes, queue, servers, LambdaArrival, LambdaService)
        elif event_type == DEPARTURE:
            departure_patients.append(patient)
            departure(clock, patient, fes, queue, servers)
        else:
            raise NameError(f"Event type {event_type} is not defined")

    return departure_patients, queue.log

def arrival(clock, patient, fes, queue, servers, LambdaArrival, LambdaService):
    # sample the time till the next patient's arrival
    next_arrival = clock+rand.expovariate(LambdaArrival)
    # schedule the next arrival
    next_patient = Patient(LambdaService, next_arrival)
    fes.put(Event(next_arrival, ARRIVAL, next_patient))
    
    code = patient.code
    # check if a server is idle to start the service
    if servers.at_least_one_idle():
        # insert patient to server and start service
        servers.put(patient)
        patient.start_service(clock)
        # schedule the departure of the client
        fes.put( Event(patient.departure_time, DEPARTURE, patient) )
    
    # if no server is idle, but the patient code is red 
    # and there is a green or yellow treatment on course, 
    # interrupts one non-priority treatment;
    elif code == RED and servers.at_least_one_non_priority():
        interrupted_patient = servers.pop()
        interruption(clock, interrupted_patient, fes, queue)
        # insert arriving patient to server and start service
        servers.put(patient)
        patient.start_service(clock)
        # schedule the departure of the client
        fes.put( Event(patient.departure_time, DEPARTURE, patient) )

    else: # otherwise, insert patient in the queue
        queue.put(patient, clock)

def departure(clock, patient, fes, queue, servers):
    # remove patient from server making the server idle
    servers.remove(patient)
    # check if there are more clients in the line, otherwise leaves server idle
    if len(queue) > 0:
        # get the first patient from the queue and start its service
        next_patient = queue.pop(clock)
        servers.put(next_patient)
        next_patient.start_service(clock)
        # schedule client's departure
        fes.put( Event(next_patient.departure_time, DEPARTURE, next_patient) )

def interruption(clock, patient, fes, queue):
    if patient.code == RED:
        print('Patient cannot be interrupted - Very urgent treatment')
        return
    # update patient in case of interruption
    patient.interruptions += 1
    patient.interruption_time = clock
    patient.remaining_time = patient.departure_time - clock
    # reset departure time
    patient.departure_time = None
    # remove scheduled patient's departure from FES
    fes.remove(patient=patient, event_type=DEPARTURE)
    # add patient to front of the queue
    queue.put(patient, clock)

def average_queue_size(queue_log):
    # computes the average length of the queue given the log of the simulation
    n = len(queue_log.time) # number of changes in the queue
    max_length = np.max(queue_log.total)
    total_time = np.max(queue_log.time)
    # stores the product of time by queue length (length equal to the list index)
    time_by_length = np.zeros(max_length+1)
    for i in range(n-1):
        length = queue_log.total[i]
        time_by_length[length] += (queue_log.time[i+1]-queue_log.time[i]) * length    
    # compute as a weighted average 
    avg_queue_size = np.sum(time_by_length) / total_time
    return avg_queue_size

def confidence_interval(values, confidence=.98):
    # computes the confidence interval for the average
    # of a particular measure where values is a list
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    ci_low, ci_upp = stats.t.interval(confidence, n-1, avg, std/n**.5)
    delta = (ci_upp-ci_low)/2
    return avg, delta

def n_simulations(n, LambdaArrival, LambdaService, NumberServers, MaxTimeCondition, confidence):
    # store the outputs of the simulations
    queue_log_sims = [None] * n
    avg_delays = np.zeros(n)
    avg_queue_sizes = np.zeros(n)

    if isinstance(LambdaService, (int, float)):
        # converts the number to the dictionary form  
        LambdaService = {
            code:LambdaService for code in [RED, YELLOW,GREEN]
        }
    
    # print simulations setup
    string = f'Servers: K={NumberServers} || '
    string += f'Arrival: \u03bba={LambdaArrival:.2f} || Service: \u03bbs=('
    for code in LambdaService:
        string += f'{code}:{LambdaService[code]:.3f};'
    string += f') || '
    print(string, end='')

    for i in range(n):
        # perform n simulations
        departure_patients, queue_log = simulation(
            LambdaArrival=LambdaArrival,
            LambdaService=LambdaService, 
            NumberServers=NumberServers,
            MaxTimeCondition=MaxTimeCondition, 
        )
        # compute the measures for each simulation
        delays = np.array([ pat.delay for pat in departure_patients ])
        avg_delays[i] = delays.mean()        
        queue_log_sims[i] = queue_log
        avg_queue_sizes[i] = average_queue_size(queue_log)                

    # compute the measures for all simulations and its confidence intervals
    delay_avg, delay_delta = confidence_interval(avg_delays, confidence=confidence)
    queue_size_avg, queue_size_delta = confidence_interval(avg_queue_sizes, confidence=confidence)

    # print simulations results
    print(f'Average delay: {delay_avg:.3f} \u00b1 {delay_delta:.3f} || ', end='')
    print(f'Average queue size: {queue_size_avg:.3f} \u00b1 {queue_size_delta:.3f}')

    # return results and queue log to be plotted
    return delay_avg, queue_size_avg, queue_log_sims

def plot_queue(queue_log, lambda_arrival, lambda_service, simulation_time, number_servers,
                plot_index, savedir='.\\figures', global_avg_queue_size=None):
    # plot queue evolution for a particular simulation given its queue log
    fig,ax = plt.subplots()
    time = np.array(queue_log.time)
    totalQ = np.array(queue_log.total)
    redQ = np.array(queue_log.RED)
    yellowQ = np.array(queue_log.YELLOW)
    greenQ = np.array(queue_log.GREEN)
    avg_queue_size = average_queue_size(queue_log)

    ax.fill_between(time, totalQ, yellowQ+redQ, facecolor='#329b32', step='pre',
                    label=f'green queue (\u03bbs={lambda_service[GREEN]:.3f})' )
    ax.fill_between(time, yellowQ+redQ, redQ, facecolor='#e6c000', step='pre',
                    label=f'yellow queue (\u03bbs={lambda_service[YELLOW]:.3f})')
    ax.fill_between(time, redQ, facecolor='#c80000', step='pre',
                    label=f'red queue (\u03bbs={lambda_service[RED]:.3f})', )
    ax.axvline(simulation_time, ls='--', c='#aaaaaa', label='end of shift', lw=.9)
    ax.axhline(avg_queue_size, ls='-.', c='#0077ff', label='avg queue size (sim)', lw=.9)
    ax.text(5, avg_queue_size+.1, f'{avg_queue_size:.2f}', fontsize=8, c='#0077ff')
    if global_avg_queue_size: # if given, allow comparison with the simulation
        ax.axhline(global_avg_queue_size, ls='-.', c='#cf4f00', label='avg queue size (global)', lw=.9)
        ax.text(5, global_avg_queue_size+.1, f'{global_avg_queue_size:.2f}', fontsize=8, c='#cf4f00')
    
    ax.set_title(f'Servers K={number_servers}; Arrival rate \u03bba={lambda_arrival}')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Queue length')
    ax.set_xlim([0,max(time)+10])
    ax.set_ylim([0, None])
    box = ax.get_position()
    ax.legend(fontsize=7.5, loc='best')
    
    filename = f'.\\la{lambda_arrival:.3f}_k{number_servers}_queue_{plot_index}.png'
    try:
        fig.savefig(f'{savedir}{filename}', format='png')
    except:
        fig.savefig(filename, format='png')

def experiment1(Lambdas_Arrival, Lambdas_Service, n, 
                NumberServers, MaxTimeCondition, confidence):
    # simulate and plot the computed measures for each pair of 
    # values in the lists of rates of arrival and service
    fig,(ax0,ax1) = plt.subplots(2)
    for l_a in Lambdas_Arrival:
        # arrays to be used on the plot
        delay_avgs = np.zeros(len(Lambdas_Service))
        queue_avgs = np.zeros(len(Lambdas_Service))

        for i,l_s in enumerate(Lambdas_Service):
            # repeat the simulation for each pair (l_a,l_s)
            delay_avg, queue_size_avg, queue_log_sims = n_simulations(
                n=n, 
                LambdaArrival=l_a, 
                LambdaService=l_s, 
                NumberServers=NumberServers, 
                MaxTimeCondition=MaxTimeCondition,
                confidence=confidence
            )
            delay_avgs[i] = delay_avg
            queue_avgs[i] = queue_size_avg

        ax0.plot(Lambdas_Service, delay_avgs, '.-', label=f'\u03bba={l_a}')
        ax1.plot(Lambdas_Service, queue_avgs, '.-', label=f'\u03bba={l_a}')
    # set the graphs
    ax0.set_ylabel('Average delay')
    ax0.legend()
    ax0.grid()
    ax1.set_ylabel('Average queue length')
    ax1.set_xlabel('Rate of service \u03bbs')
    ax1.legend()
    ax1.grid()
    plt.show()

if __name__ == '__main__':

    # Simulation conditions
    NUMBER_SERVERS = 1
    MAX_SIMULATION_TIME = 8*60 # an eight-hour shift
    n = 1500           # number of simulations
    conf = 0.98        # confidence level
    print(f'Experiment 1 - comparison between \u03bba and \u03bbs (takes more time - graph already added to the report)')
    print(f'Experiment 2 - single run for an only set of arrival and service rates')
    experiment = int(input('Select experiment: '))

    # Experiment 1 - comparison between lambdas
    if experiment == 1:
        Lambdas_Arrival = np.arange(.5,2.1,.5)
        Lambdas_Service = np.arange(.4,2.1,.2)
        # takes more time - graph already added to the report
        experiment1(Lambdas_Arrival, Lambdas_Service, n=n, NumberServers=NUMBER_SERVERS, 
            MaxTimeCondition=MAX_SIMULATION_TIME, confidence=conf)

    # Experiment 2 - single run for an only set of arrival and service rates 
    elif  experiment == 2:
        LAMBDA_ARRIVAL = 0.7
        LAMBDA_SERVICE = {RED:1/10, YELLOW:1/6, GREEN:1/3}
        delay_avg, queue_size_avg, queue_log_sims = n_simulations(
            n, LambdaArrival=LAMBDA_ARRIVAL, LambdaService=LAMBDA_SERVICE, 
            NumberServers=NUMBER_SERVERS, MaxTimeCondition=MAX_SIMULATION_TIME,
            confidence=conf
        )

        # plot random executions of the simulation
        n_plots = int(input('Number of simulations to visualize: '))
        queue_logs_plot = rand.sample(queue_log_sims, k=n_plots)
        for i,qlog in enumerate(queue_logs_plot):
            plot_queue(
                qlog, lambda_arrival=LAMBDA_ARRIVAL, lambda_service=LAMBDA_SERVICE, 
                simulation_time=MAX_SIMULATION_TIME, number_servers=NUMBER_SERVERS,
                global_avg_queue_size=queue_size_avg, plot_index=i, 
            )