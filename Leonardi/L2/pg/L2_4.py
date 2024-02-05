import random
from queue import PriorityQueue
from tqdm import tqdm

# the following are the arrival rates, that can be tuned to compare different results
arrival_lambdas = [5,7,9,10,12,15]

# tunable parameters: they represent the service rate for each color code of the patients
SERVICE_RED = 2.0
SERVICE_YELLOW = 1.0
SERVICE_GREEN = 0.6
# clearly, the time spent for serving a patient with red code is higher than the one for 
# a yellow code or a green code

SIM_TIME = 2_00
# the higher is the simulation time, the higher would be the number of events 

# Maximum capacity of the waiting line
MAX_QUEUE_CAPACITY = 50
3,3,3,3,1,3,3,1
tot_studios = 1 # number of studio(s)

FES = PriorityQueue()
waiting_line = []


class Client:
    def __init__(self, time):
        self.patient_type = None
        self.arrival_time = time
        self.departure_time = 0
        self.is_served = False

    def set_departure_time(self, departure_time):
      self.departure_time = departure_time

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
        # the patients are ordered by arrival time

    def __str__(self):
      return f'The patient of type {self.patient_type} is arrived at {self.arrival_time} and departs at {self.departure_time}'
    # this method is useful to visualize what happens:
    # if the patient is arrived, print at what time did he arrive and at what time would he leave

# references and initializations
time = 0
 
clients_served = PriorityQueue()  # these are the clients which are served in a studio
FES = PriorityQueue() # this is the Future Event Set
waiting_line = [] # the waiting line is just a list, not a queue

free_studios = tot_studios # initially, the number of free studios is equal to the total number of studios
reds = [] # these are the patients with red code
yellows = [] # these are the patients with yellow code
greens = [] # these are the patients associated with green code
users_in_system = 0 # these are the total number of patients in the Emergency Room considering both the served patients and the ones in the waiting line
events = [] # this list would take trace of what the patients do

def arrival(time, lambda_):

    global free_studios
    global users_in_system
    dropped_clients = 0 # these are the number of patients which cannot enter anymore 
    # compute the inter-arrival time for next patient
    inter_arrival = random.expovariate(lambda_)

    # firstly, we assign a type to each new patient basing on probabilities given by assignment
    random_value = random.random()
    if random_value < 1/6:
        client_type = 'RED'
        client = Client(time)
        client.patient_type = client_type

        reds.append(client)
        # print(reds)

    elif random_value < 1/2:
        client_type = 'YELLOW'
        client = Client(time)
        client.patient_type = client_type 

        yellows.append(client)
        # print(yellows)

    else:
        client_type = 'GREEN'
        client = Client(time)
        client.patient_type = client_type

        greens.append(client)
        # print(greens)
    
    # notice that each time a patient is created, he is also added to a list of patiemnts with the same color type

    # schedule a next arrival  
    FES.put((time + inter_arrival, 'arrival'))
    # print(FES.queue) # just a check
    events.append(client) 
    users_in_system+=1

    if(client.patient_type == 'YELLOW'):
      service_time = random.expovariate(SERVICE_YELLOW)
    elif(client.patient_type == 'GREEN'):
      service_time = random.expovariate(SERVICE_GREEN)
    elif(client.patient_type == 'RED'):
      service_time = random.expovariate(SERVICE_RED)

    # each time that a patient arrives, we firstly compute the next arrival and then we compute the service time 
    # then we check the color code of the patient just arrived

    if(client.patient_type == 'YELLOW' or client.patient_type == 'GREEN'): 
      # if the color code of the patient is not red (high risk), then we do the following
      
      # CASE 1) if the waiting line is full and there are no free studios, then the patient is dropped and the related variable is incremented
      if len(waiting_line) == MAX_QUEUE_CAPACITY and free_studios==0:
        dropped_clients += 1
        # print(dropped_clients)
        FES.put((time, 'DROP'))
        # print(FES.queue)
      
      # CASE 2) if the waiting line is not empty but neither full and if there are some free studios, then we check the patients waiting in the waiting line
      elif (len(waiting_line) < MAX_QUEUE_CAPACITY and len(waiting_line) > 0) and free_studios > 0:
        only_greens = 0

        # print(waiting_line)

        # for each patient in the waiting line, if he is with green code, then we increment the relative counter 
        for i, waiting_client in enumerate(waiting_line):
          if waiting_client.patient_type == 'GREEN':
              only_greens +=1

        # CASE A) if the just arrived patient is green:
        # -> if the counter is equal to the length of the waiting line, then all the people in line are green, so the just arrived client is put in line with the others as last
        # -> if there's a yellow code in the waiting line, then the green patient is put in the last seat of the waiting line to leave priority to the yellow one
        if only_greens == len(waiting_line) and client.patient_type == 'GREEN':
          waiting_line.append(client)
          FES.put((time, 'WAIT'))
          users_in_system += 1
        elif only_greens < len(waiting_line) and client.patient_type == 'GREEN':
          waiting_line.append(client)
          FES.put((time, 'WAIT'))

        # CASE B) if the the just arrived patient is yellow code:
        # -> if the counter is equal to the length of the waiting line, then all the waiting people are green so the yellow one is served
        # -> if there is another yellow patient in the line then the just arrived patient is put after the already present yellow patient
        elif only_greens == len(waiting_line) and client.patient_type == 'YELLOW':
            client.is_served = True
            clients_served.put( client)
            free_studios -= 1
            client.set_departure_time(time + service_time)
            FES.put((time + service_time, 'departure'))
            users_in_system += 1
        elif only_greens < len(waiting_line) and client.patient_type == 'YELLOW':
            waiting_line.insert(1, client)
           
      # CASE 3 and 4) if the waiting line is empty but there are some free studios (possibly all of them are free), then the just arrived patient can be served immediately
      # so, the departure time is scheduled, the free studio is occupied and the number of patients in the system increases
      elif len(waiting_line) == 0 and free_studios >= 0:
        client.is_served = True
        clients_served.put( client)
        free_studios -= 1
        client.set_departure_time(time + service_time)
        FES.put((time + service_time, 'departure'))
        users_in_system += 1

        # print(clients_served)

      # after the departure, the client stops to be served and the patients in the system decrease
      client.is_served = False 
      users_in_system -= 1
      clients_served.get()

    # instead if the just arrived patient is a red code
    elif client.patient_type == 'RED':
      # CASE A) if there are some free studios available, then one of them is occupied by the just arrived patient, the departure time is set and the event is created
      if free_studios > 0:
        free_studios -= 1
        client.is_served = True
        clients_served.put(client)
        client.set_departure_time(time + service_time)
        events.append(client)
        FES.put((time + service_time, 'departure'))
        print(clients_served)

      # CASE B) if there is no free studio available, one of the client present in one of the studios is stopped in order to leave the priority to the red code
      else:
        stop_client = clients_served.get()
        stop_client.is_served = False
        FES.put((client.arrival_time, 'STOP')) # the client stopped is put in the fes to take trace of the time of the event, which is exactly the time in which the red patient arrives

        # then the red code patient is served, the fes is fulfilled and the number of users in the system increases
        clients_served.put(client)
        client.is_served = True
        client.set_departure_time(time + service_time)
        FES.put((time + service_time, 'departure'))
        events.append(client)
        users_in_system += 1

        # after the departure of the red code, the stopped patient can continue to be served, so the new departure time is set basing on how much time of service the stopped patient needs
        new_departure = client.departure_time + service_time - (client.arrival_time - stop_client.arrival_time)
        stop_client.set_departure_time(new_departure)
        clients_served.put(stop_client)
        FES.put((new_departure, 'departure'))
        events.append(stop_client)
        # another event is created but the counter of the patients present in the system is not increasing since this patient was already counted in the system before

      users_in_system -= 1
      free_studios += 1
      clients_served.get()
      client.is_served = False
      # then, the studio is put back free for the next service

def departure(time):
    global free_studios
    global users_in_system
    users_in_system -= 1

    # print(waiting_line)

    # if the waiting line is not full nor empty, and there are some studios free, then the event about departure is set:
    # the first patient in the row is take under consideration to be served
    if (len(waiting_line) < MAX_QUEUE_CAPACITY and len(waiting_line) > 0) and free_studios > 0:
      client_now = waiting_line[0]

      # the service time is scheduled and so the departure time, basing on the color code

      if(client_now.patient_type == 'YELLOW'):
        service_time = random.expovariate(SERVICE_YELLOW)
      elif(client_now.patient_type == 'GREEN'):
        service_time = random.expovariate(SERVICE_GREEN)
      elif(client_now.patient_type == 'RED'):
        service_time = random.expovariate(SERVICE_RED)

      client_now.set_departure_time(time + service_time)
      users_in_system -= 1
      free_studios -= 1
      FES.put((time + service_time, 'departure'))
      events.append(client_now)
      # the event is created and after the departure, the studio is back free for the next service

    free_studios += 1
    
# after the initialization already present above, we can start the simulation
FES.put((time, "arrival"))
lambda_values = [5, 7, 9, 12, 17]

# Initialize time and the first arrival event
# time = 0
FES.put((time, "arrival"))

# Rest of your code

for lambda_ in tqdm(lambda_values):
    while time < SIM_TIME:
        if FES.empty():  # Check if the event queue is empty
            print('FES is empty')
            break

        # Retrieve the next event and update time
        (time, event_type) = FES.get()

        if event_type == "arrival":
            arrival(time, lambda_)
        elif event_type == "departure":
            departure(time)