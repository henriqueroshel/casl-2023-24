# COMPUTER-AIDED SIMULATIONS LAB
# LAB 1 - QUEUING SYSTEM 

import numpy as np
from random import expovariate, choice
from collections import namedtuple

# Simulation conditions
NUMBER_SERVERS = 10
WAITING_LINE_SIZE = 1000
LAMBDA_SERVICE_DEFAULT = 1
LAMBDAS_ARRIVAL = {5,7,9,10,12,15}
LAMBDA_ARRIVAL_DEFAULT = 10

# Default stop conditions constants
MAX_SIMULATION_TIME = 8*60 # an eight-hour shift

# other constants
ARRIVAL, DEPARTURE = 'arrival', 'departure'

# auxiliary objects
Event = namedtuple('Event', ['time', 'type'])
Client = namedtuple('Client', ['event_type', 'arrival_time'])

# Future-Event Set as list as specified
def put(FES, event):
    FES.append(event)
def pop(FES):
    # pop next event (lowest time) from the FES
    next_event = min(FES, key=lambda e:e.time)
    next_event_index = FES.index( next_event )
    return FES.pop( next_event_index )

def stop_condition(time, time_condition=None, arrivals_condition=None):
    # check the stop condition of the simulation;
    # can be toggled to stop according to:
    # - time_condition: the simulation stops when time > time_condition
    # - arrivals_condition: the simulation stops when arrivals > arrivals_condition
    # return True if at least one stop condition is satisfied

    stopcond = False
    
    if time_condition:
        stopcond = stopcond or (time>time_condition)
    if arrivals_condition:
        global arrivals
        stopcond = stopcond or (arrivals>arrivals_condition)
    
    return stopcond


def arrival(time, FES, queue, LambdaArrival, LambdaService):
    global users, idle_servers
    global delays, dropped
    
    # sample the time until the next arrival
    inter_arrival = expovariate(LambdaArrival)
    # schedule the next arrival
    put(FES, Event(time+inter_arrival, ARRIVAL))
    
    # managing the event
    # create a record for the client
    client = Client(ARRIVAL, time)

    # drop client if queue at maximum capacity 
    if users >= WAITING_LINE_SIZE:
        dropped += 1

    else:
        # check if a server is idle to start the service
        if idle_servers > 0:
            # make a server busy
            idle_servers -= 1
            # register delay equals to zero
            delays.append(0)           
            # sample the service time 
            service_time = expovariate(LambdaService)
            # schedule the departure of the client
            put(FES, Event(time+service_time, DEPARTURE))
        else:
            # if no server is idle, update state variable 
            users += 1
            # insert client in the queue
            queue.append(client)


def departure(time, FES, queue, LambdaService):
    global users, idle_servers
    global delays

    # check if there are more clients in the line
    if users > 0:
        # get the first client from the queue
        client = queue.pop(0)
        # update state variable
        users -= 1
        # register client delay
        delays.append(time - client.arrival_time)
        # sample the service time 
        service_time = expovariate(LambdaService)
        # schedule when the client's departure
        put(FES, Event(time+service_time, DEPARTURE))
    else:
        # if there are no clients in the line, the server becomes idle
        idle_servers += 1


def simulation( LambdaArrival=None, 
                LambdaService=None, 
                MaxTimeCondition=None, 
                MaxArrivalsCondition=None
                ):
    
    global users, arrivals, idle_servers, dropped


    # define default simulation conditions
    if not LambdaArrival:
        LambdaArrival = LAMBDA_ARRIVAL_DEFAULT
    if not LambdaService:
        LambdaService = LAMBDA_SERVICE_DEFAULT

    print('\n---------------------------------------------')
    print("SIMULATION CONDITIONS")
    print(f'Arrival rate: {LambdaArrival}')
    print(f'Service rate: {LambdaService}\n')

    print('STOP CONDITION(S):')    
    if MaxArrivalsCondition:
        print(f'Max arrivals: {MaxArrivalsCondition}')
    if MaxTimeCondition:
        print(f'Max simulation time: {MaxTimeCondition}')
    if not (MaxTimeCondition or MaxArrivalsCondition):
        MaxTimeCondition = MAX_SIMULATION_TIME
        print(f'Max simulation time: {MaxTimeCondition}')

    # initialize FIFO queue
    queue = [] 

    # simulation clock
    time = 0
    # list of Event objects as (time, type) in a priority queue
    FES = []
    # schedule first arrival at t=0
    put(FES, Event(0, ARRIVAL))
    
    # main loop of simulation
    while not stop_condition(time, time_condition=MaxTimeCondition, arrivals_condition=MaxArrivalsCondition):
        # select next event in the FES with the lowest time
        time, event_type = pop(FES)

        if event_type == ARRIVAL:
            arrivals += 1
            arrival(time, FES, queue, LambdaArrival, LambdaService)
        elif event_type == DEPARTURE:
            departure(time, FES, queue, LambdaService)
        else:
            raise NameError(f"Event type {event_type} is not defined")

    print("\nSIMULATION RESULTS")
    # computation of the selected measures
    avg_delay = np.mean(delays)
    drop_probability = dropped / arrivals
    print(f'Average delay time: {avg_delay:.4f}')
    print(f'Dropping probability: {drop_probability:.2%}')
    print('---------------------------------------------')


        
if __name__ == '__main__':

    print(f'Hours of simulation: \n(default: {MAX_SIMULATION_TIME//60} hours) (blank to skip)')
    simulation_time = input()
    if simulation_time:
        simulation_time = int(simulation_time)*60
    print(f'Maximum number of arrivals: \n(default: None) (blank to skip)')
    max_arrivals = input()
    if max_arrivals:
        max_arrivals = int(max_arrivals)
    

    for LambdaArrival in LAMBDAS_ARRIVAL:

        # state variable - # of idle servers in the system
        idle_servers = NUMBER_SERVERS
        # state variable - # of clients in the queue
        users = 0
        # count of arrivals of clients in the system
        arrivals = 0
        # clients who left due to waiting line at maximum capacity
        dropped = 0 
        # delay of each client to compute the average delay
        # delay = time between arrival and start of service
        delays = []

        simulation(
            LambdaArrival=LambdaArrival, 
            MaxTimeCondition=simulation_time,
            MaxArrivalsCondition=max_arrivals
        )