import random
import numpy as np
import matplotlib.pyplot as plt


class Client:
    def __init__(self, arrival, priority):
        self.arr_time = arrival
        self.dep_time = None
        self.pr = priority
    def __repr__(self) -> str:
        if self.dep_time == None:
            return f'(arr: {self.arr_time:.2f}, dep: None, priority: {self.pr})'
        else:
            return f'(arr: {self.arr_time:.2f}, dep: {self.dep_time:.2f}, priority: {self.pr})'

class Event:
    def __init__(self, type, time, priority):
        self.type = type
        self.time = time
        self.pr = priority
    def __repr__(self):
        return f'{self.type} [{self.pr}] at {self.time:.2f}'

class Metrics:
    def __init__(self):
        self.arrivals = 0
        self.dep = 0
        self.delay = 0

def create_arrival(candidates, probs, time):
    draw = np.random.choice(candidates, replace = False, p = probs)
        
    if draw == 0:
        event = Event('arrival', time, 0)
    elif draw == 1:
        event = Event('arrival', time, 1)
    elif draw == 2:
        event = Event('arrival', time, 2)
    return event

def check_for_no_red(queue):
    flag = True

    for event in queue:
        if event.pr == 0:
            flag = False
            break
    return flag

SIM_TIME = 10000
lambdas_arr = [i for i in range(1, 11)] # different arrival rates
lambda_red_ser = 1 # service rate for red patients
lambda_yellow_ser = 1 # service rate for yellow patients
lambda_green_ser = 1 # service rate for green patients
priorities = (0, 1, 2) # patients priorities: 0->red, 1->yellow, 2->green
probs = [1/6, 1/3, 1/2] # probability of coming patients
prt_probs = [probs[i]/sum(probs) for i in range(len(probs))] # divide probabilites in order to be sure that their sum is equal to 1

# dictionaries to storage average delay for different types of patients using different arrival rate
r_avg_delay = {}
y_avg_delay = {}
g_avg_delay = {}

for lambda_arr in lambdas_arr:
        
    CURRENT_TIME = 0
    interruption = False
    remain_service_time = 0 # the remain time of the interrupted patient
    interrupted_service = -1 # indicating the type of interrupted patient
    current_service = -1 # indicating status of the medical team: -1 -> no one is receiving service, 0 -> a red is ..., 1 -> a yellow is ..., 2 -> a green is ...

    #different metircs for different types of patients
    r_metrics = Metrics()
    y_metrics = Metrics()
    g_metrics = Metrics()

    queue = []
    FES = [create_arrival(priorities, prt_probs, 0)]

    while len(FES) > 0 and CURRENT_TIME < SIM_TIME:
                    
        FES.sort(key = lambda x: (x.time, x.pr), reverse=True)                
        current_event = FES.pop()

        if current_event.type == 'arrival':

            # scheduling the next arrival
            inter_arrival = random.expovariate(1/lambda_arr)
            next_arrival = create_arrival(priorities, prt_probs, CURRENT_TIME + inter_arrival)
            FES.append(next_arrival)
            
            # handling the current arrival
            current_client =  Client(CURRENT_TIME, current_event.pr)

            if current_client.pr == 0: # if the arrival is red, it interrupts the current service unless the current service is red
                r_metrics.arrivals += 1
                if current_service != 0: # if the current service isn't red
                    service_time = random.expovariate(1/lambda_red_ser)
                    current_client.dep_time = CURRENT_TIME + service_time

                    if current_service != -1: # if the current service is yellow or green, the an interruption happens
                        new_FES = []
                        for event in FES:
                            if event.type == 'arrival':
                                new_FES.append(event)
                            elif event.type == 'departure':
                                interruption = True
                                remain_service_time = event.time - CURRENT_TIME
                                interrupted_service = event.pr
                        FES = new_FES

                    dep = Event('departure', current_client.dep_time, current_client.pr)
                    FES.append(dep)
                    current_service = 0
                else:
                    queue.insert(0, current_client)
            
            else: # if the arrival is yellow or green
                if current_client.pr == 1:
                    y_metrics.arrivals += 1
                elif current_client.pr == 2:
                    g_metrics.arrivals += 1
             
                if len(queue) == 0 and current_service == -1: # if the queue is empty and no one is receiving service

                    if current_client.pr == 1:
                        service_time = random.expovariate(1/lambda_yellow_ser)
                    elif current_client.pr == 2:
                        service_time = random.expovariate(1/lambda_green_ser)

                    current_client.dep_time = CURRENT_TIME + service_time
                    dep = Event('departure', current_client.dep_time, current_client.pr)
                    FES.append(dep)
                    current_serving = current_client.pr
                else: # otherwise it must enter the queue
                    queue.insert(0, current_client)
            
            CURRENT_TIME += inter_arrival

        elif current_event.type == 'departure':
            
            if current_event.pr == 0:
                r_metrics.dep += 1
                r_metrics.delay += CURRENT_TIME - current_event.time
            elif current_event.pr == 1:
                y_metrics.dep += 1
                y_metrics.delay += CURRENT_TIME - current_event.time
            elif current_event.pr == 2:
                g_metrics.dep += 1
                g_metrics.delay += CURRENT_TIME - current_event.time

            current_service = -1

            if interruption and check_for_no_red(queue):
                dep = Event('departure', CURRENT_TIME + remain_service_time, interrupted_service)
                FES.append(dep)
                current_service = interrupted_service
                interruption = False
                remain_service_time = 0
                interrupted_service = -1
            else:
                if len(queue) > 0:
                    queue.sort(key = lambda x: (x.pr, x.arr_time), reverse=True)
                    client = queue.pop()

                    if client.pr == 1:
                        service_time = random.expovariate(1/lambda_yellow_ser)
                    elif client.pr == 2:
                        service_time = random.expovariate(1/lambda_green_ser)

                    FES.append(Event('departure', CURRENT_TIME + service_time, client.pr))
                    current_service = client.pr


    r_avg_d = r_metrics.delay/(r_metrics.dep+0.00001)
    y_avg_d = y_metrics.delay/(y_metrics.dep+0.00001)
    g_avg_d = g_metrics.delay/(g_metrics.dep+0.00001)

    r_avg_delay[lambda_arr] = r_avg_d
    y_avg_delay[lambda_arr] = y_avg_d
    g_avg_delay[lambda_arr] = g_avg_d


plt.scatter(r_avg_delay.keys(), r_avg_delay.values(), label=f'red, lambda: {lambda_red_ser}', c='r')
plt.plot(r_avg_delay.keys(), r_avg_delay.values(), linestyle='dashed', c='r')

plt.scatter(y_avg_delay.keys(), y_avg_delay.values(), label=f'yellow, lambda: {lambda_yellow_ser}', c='yellow')
plt.plot(y_avg_delay.keys(), y_avg_delay.values(), linestyle='dashed', c='yellow')

plt.scatter(g_avg_delay.keys(), g_avg_delay.values(), label=f'green, lambda: {lambda_green_ser}', c='g')
plt.plot(g_avg_delay.keys(), g_avg_delay.values(), linestyle='dashed', c='g')

plt.legend()
plt.xlabel('Lambda')
plt.ylabel('Delay')
plt.show()