import numpy as np
from numpy import random
import itertools
import matplotlib.pyplot as plt
from scipy import stats

# As it will be needed to occasionally remove elements from the middle of the FES, a Priority Queue isn't enough to implement it, we need something more complex
# The structure of an element in the FES is supposed to be (time, type, id of customer, code of customer)
class ExtendedPriorityQueue :
    def __init__(self) -> None:
        self.queue = list()
    
    def sort(self) :
        self.queue.sort(key = lambda elem : elem[0])
    
    def put(self, elem) :
        self.queue.append(elem)
        self.sort()
    
    def get(self) :
        if self.queue == list() :
            return None
        return self.queue.pop(0)
    
    # Remove an event based on the id of the customer
    def remove(self,id) :
        list_ids = list([elem[2] for elem in self.queue])
        self.queue.pop(list_ids.index(id))


# Contains all the important attributes of an arriving customer
class Client :
    id_iter = itertools.count()
    def __init__(self, arrival_time) -> None:
        self.id = next(self.id_iter)
        self.arrival_time = arrival_time
        # Random attribution of a code based on the given probabilities
        self.code = random.choice(['Red','Yellow','Yellow','Green','Green','Green'], size=1)[0]

        # The service time is an attribute of the customer, as he could be served in multiple times due to priorities --> Its mean is arbitrarily fixed to 10
        self.service_time = random.exponential(1/10, size=1)
        # Indicates if the customer is currently being taken care of by a medical team
        self.being_served = False
        self.start_of_service = -1


class Measure :
    def __init__(self, Narr = 0, Ndep = 0, NredDep = 0, NyellowDep = 0, NgreenDep = 0,
                  Nreds = 0, Nyellows = 0, Ngreens = 0, NAverageUser = 0, oldTimeEvent = 0, 
                  delay = 0, redDelay = 0, yellowDelay = 0, greenDelay = 0) -> None:
        self.arrivals = Narr
        self.departures = Ndep
        self.red_departures = NredDep
        self.yellow_departures = NyellowDep
        self.green_departures = NgreenDep
        self.reds = Nreds
        self.yellows = Nyellows
        self.greens = Ngreens
        self.users_with_time = NAverageUser
        self.old_time = oldTimeEvent
        self.delay = delay 
        self.red_delay = redDelay
        self.yellow_delay = yellowDelay
        self.green_delay = greenDelay


# Handles the event of arrival of a customer
def arrival(time, FES, queues, K, LAMBDA) :
    global users
    global data

    # Schedule next arrival of customer with unknown ID and code
    inter_arrival_time = random.exponential(1/LAMBDA)
    FES.put((time + inter_arrival_time, "arrival", -1, '_'))

    client = Client(time)

    # Stats
    data.arrivals += 1
    data.users_with_time += users * (time - data.old_time)
    data.old_time = time
    
    if client.code == 'Red' :
        data.reds += 1
    elif client.code == 'Yellow' :
        data.yellows += 1
    else :
        data.greens += 1
    
    # Queues is a dict containing the queues for each code
    queues[client.code].append(client)

    users += 1
    
    # If at least one team can take care of the customer, it is done immediately 
    if users <= K :
        FES.put((time + client.service_time, "departure", client.id, client.code))
        client.being_served = True
        client.start_of_service = time
        return
    
    if client.code == 'Red' :
        available_place = False
        # If a green customer is being served, the current red can take his place
        first_green = queues['Green'][0] if len(queues['Green']) > 0 else None
        if first_green is not None and first_green.being_served :
            # Compute new service time for the green customer
            new_service_time = first_green.service_time - (time - first_green.start_of_service)
            first_green.service_time = new_service_time
            # Interrupt service of customer
            first_green.being_served = False
            first_green.start_of_service = -1
            # Unschedule departure of customer, who goes back at the front of its queue
            FES.remove(first_green.id)
            available_place = True

        # If no green customer is being served, then we can check if a yellow one is being served
        first_yellow = queues['Yellow'][0] if len(queues['Yellow']) > 0 else None
        if not available_place and first_yellow is not None and first_yellow.being_served :
            new_service_time = first_yellow.service_time - (time - first_yellow.start_of_service)
            first_yellow.service_time = new_service_time
            first_yellow.being_served = False
            first_yellow.start_of_service = -1
            FES.remove(first_yellow.id)
            available_place = True  

        # If one of the above conditions as respected, the red client is immediately taken care of, and his departure is scheduled
        if available_place :
            FES.put((time + client.service_time, "departure", client.id, client.code))
            client.being_served = True
            client.start_of_service = time


# Handles the event of departure of a customer (id and code are arguments in order to find the specific customer, for the stats and for interruptions)
def departure(time, FES, queues, K, id, code) :
    global users 
    global data

    # Stats
    data.departures += 1
    data.users_with_time += users * (time - data.old_time)
    data.old_time = time

    # Collect stats for the served customer then remove him from queue
    queue = queues[code]
    index = next(i for i, elem in enumerate(queue) if elem.id == id)
    client = queue[index]
    data.delay += (time - client.arrival_time)
    if code == 'Red' :
        data.red_departures += 1
        data.red_delay += (time - client.arrival_time)
    elif code == 'Yellow' :
        data.yellow_departures += 1
        data.yellow_delay += (time - client.arrival_time)
    else :
        data.green_departures += 1
        data.green_delay += (time - client.arrival_time)
    queue.remove(client)
    users -= 1

    # Get new client to be served (if there are more people in queue than available teams, there will necessarily be a customer to serve)
    if users > K - 1 :
        client = None
        while True :
            for c in queues['Red'] :
                if not c.being_served :
                    client = c
                    break
            if client != None :
                break
            for c in queues['Yellow'] :
                if not c.being_served :
                    client = c
                    break
            if client != None :
                break
            for c in queues['Green'] :
                if not c.being_served :
                    client = c
                    break
            if client != None :
                break
        FES.put((time + client.service_time, "departure", client.id, client.code))
        client.being_served = True
        client.start_of_service = time


def run_simulation(K,Lambda,sim_time) :
    time = 0

    FES = ExtendedPriorityQueue()
    queues = dict()
    queues['Red'] = list()
    queues['Yellow'] = list()
    queues['Green'] = list()
    
    FES.put((0, "arrival", -1, '_'))

    while time < sim_time : 
        (time, event_type, id_cust, code_cust) = FES.get()
        if event_type == "arrival" :
            arrival(time, FES, queues, K, Lambda)
        elif event_type == "departure" :
            departure(time, FES, queues, K, id_cust, code_cust)

    global data 
    results = dict()
    # Global stats
    results['average_delay'] = (data.delay / data.departures if data.departures != 0 else np.inf)[0]
    results['average_number_cust'] = (data.users_with_time / time)[0]

    # Code-specific stats
    results['average_delay_red'] = (data.red_delay / data.red_departures if data.red_departures != 0 else [np.inf])[0]
    results['average_delay_yellow'] = (data.yellow_delay / data.yellow_departures if data.yellow_departures != 0 else [np.inf])[0]
    results['average_delay_green'] = (data.green_delay / data.green_departures if data.green_departures != 0 else [np.inf])[0]

    results['unserved_reds'] = data.reds - data.red_departures
    results['unserved_yellows'] = data.yellows - data.yellow_departures
    results['unserved_greens'] = data.greens - data.green_departures
    
    return results


def compute_conf_interval_and_accuracy(sample, conf_level) :
    if np.inf in sample :
        return [np.inf, np.inf], 1.

    mean = np.mean(sample)
    sd = np.std(sample, ddof=1)
    alpha = 1 - conf_level
    n = len(sample)
    if n <= 30 :
        quantile = stats.t(df=n-1).ppf(1 - alpha/2)
    else :
        quantile = stats.norm().ppf(1 - alpha/2)
    
    delta = quantile*sd/np.sqrt(n)
    err = delta/mean if delta/mean <= 1. else 1.
    accuracy = 1 - err

    return [mean - delta, mean + delta], accuracy


# Runs the simulation and collect data until all 95% confidence intervals for the measures are at least acc-accurate
def get_accurate_measures(K, lambda_, sim_time, acc) :
    global users, data
    accuracy = 0
    measures = dict()
    intervals = dict()
    while accuracy < acc :
        users = 0
        data = Measure()
        res = run_simulation(K, lambda_, sim_time)
        min_acc = 1.0
        if measures == dict() :
            for key, value in res.items() :
                measures[key] = list([value])
                min_acc = 0.0
        else :
            for key, value in res.items() :
                measures[key].append(value)       
                interval, acc_spec = compute_conf_interval_and_accuracy(measures[key], 0.95)
                intervals[key] = interval
                if acc_spec < min_acc :
                    min_acc = acc_spec
        accuracy = min_acc
    return intervals


# Run simulations and collect intervals for all values of customer arrival rate in arguments
def loop_over_arrival_rate(lambda_range) :
    results = dict({'average_delay': [],
                    'average_number_cust': [],
                    'average_delay_red': [],
                    'average_delay_yellow': [],
                    'average_delay_green': [],
                    'unserved_reds': [],
                    'unserved_yellows': [],
                    'unserved_greens': []})
    for lambda_ in lambda_range :
        intervals = get_accurate_measures(1,lambda_,1000,0.5)
        for key, interval in intervals.items() :
            results[key].append(interval)
    
    return results


# Display confidence intervals in organized graphs
def plot_results(lambda_range, final_data) :
    fig, axs = plt.subplots(2,2, figsize=(13,13))
    fig.suptitle("Global and specific statistics for K = 1 medical team")

    # Graph 1 : Average global delay
    avg_glob_delay = np.array(final_data['average_delay'])
    axs[0,0].plot(lambda_range, avg_glob_delay[:,0] + (avg_glob_delay[:,1] - avg_glob_delay[:,0])/2, '-o')
    axs[0,0].fill_between(lambda_range, avg_glob_delay[:,0], avg_glob_delay[:,1], alpha = 0.2)
    axs[0,0].set_xlabel('Arrival rate')
    axs[0,0].set_ylabel('Delay')
    axs[0,0].set_title('Average time of presence in queue')

    # Graph 2 : Average number of users in queue
    avg_custs = np.array(final_data['average_number_cust'])
    axs[0,1].plot(lambda_range, avg_custs[:,0] + (avg_custs[:,1] - avg_custs[:,0])/2, '-o')
    axs[0,1].fill_between(lambda_range, avg_custs[:,0], avg_custs[:,1], alpha = 0.2)
    axs[0,1].set_xlabel('Arrival rate')
    axs[0,1].set_ylabel('Number of customers')
    axs[0,1].set_title('Average number of customers in queue')

    # Graph 3 : Average delay for each emergency code
    avg_red_delay = np.array(final_data['average_delay_red'])
    avg_yellow_delay = np.array(final_data['average_delay_yellow'])
    avg_green_delay = np.array(final_data['average_delay_green'])
    axs[1,0].set_yscale('log')
    axs[1,0].plot(lambda_range, avg_red_delay[:,0] + (avg_red_delay[:,1] - avg_red_delay[:,0])/2, '-o', color='red', label='Red codes')
    axs[1,0].fill_between(lambda_range, avg_red_delay[:,0], avg_red_delay[:,1], color='red', alpha=0.2)
    axs[1,0].plot(lambda_range, avg_yellow_delay[:,0] + (avg_yellow_delay[:,1] - avg_yellow_delay[:,0])/2, '-o', color='yellow', label='Yellow codes')
    axs[1,0].fill_between(lambda_range, avg_yellow_delay[:,0], avg_yellow_delay[:,1], color='orange', alpha=0.2)
    axs[1,0].plot(lambda_range, avg_green_delay[:,0] + (avg_green_delay[:,1] - avg_green_delay[:,0])/2, '-o', color='green', label='Green codes')
    axs[1,0].fill_between(lambda_range, avg_green_delay[:,0], avg_green_delay[:,1], color='green', alpha=0.2)
    axs[1,0].legend()
    axs[1,0].set_xlabel('Arrival rate')
    axs[1,0].set_ylabel('Delay')
    axs[1,0].set_title('Average time of presence \n for each emergency code')

    # Graph 4 : Number of customers still waiting at the end of simulation
    remaining_reds = np.array(final_data['unserved_reds'])
    remaining_yellows = np.array(final_data['unserved_yellows'])
    remaining_greens = np.array(final_data['unserved_greens'])
    axs[1,1].set_yscale('log')
    axs[1,1].plot(lambda_range, remaining_reds[:,0] + (remaining_reds[:,1] - remaining_reds[:,0])/2, '-o', color='red', label='Red codes')
    axs[1,1].fill_between(lambda_range, remaining_reds[:,0], remaining_reds[:,1], color='red', alpha=0.2)
    axs[1,1].plot(lambda_range, remaining_yellows[:,0] + (remaining_yellows[:,1] - remaining_yellows[:,0])/2, '-o', color='yellow', label='Yellow codes')
    axs[1,1].fill_between(lambda_range, remaining_yellows[:,0], remaining_yellows[:,1], color='orange', alpha=0.2)
    axs[1,1].plot(lambda_range, remaining_greens[:,0] + (remaining_greens[:,1] - remaining_greens[:,0])/2, '-o', color='green', label='Green codes')
    axs[1,1].fill_between(lambda_range, remaining_greens[:,0], remaining_greens[:,1], color='green', alpha=0.2)
    axs[1,1].legend()
    axs[1,1].set_xlabel('Arrival rate')
    axs[1,1].set_ylabel('Number of customers')
    axs[1,1].set_title('Average number of still waiting \n customers at the end of simulation')

    plt.show()


# Global variables
users = 0
data = Measure()

def main() :
    lambda_range = range(1,20,3)
    final_data = loop_over_arrival_rate(lambda_range)
    plot_results(lambda_range, final_data)

if __name__ == '__main__' :
    main()