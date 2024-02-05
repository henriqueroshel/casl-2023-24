import numpy as np

# Constants
RED = 'red'
YELLOW = 'yellow'
GREEN = 'green'
SERVICE_TIMES = {RED: 20, YELLOW: 30, GREEN: 40}  # mean service times in minutes
ARRIVAL_RATES = {RED: 1/6, YELLOW: 1/3, GREEN: 1/2}  # fractions of arrival
SIMULATION_TIME = 240  # simulate for 4 hours (in minutes)

# Function to generate next arrival time
def generate_next_arrival(rate):
    return -np.log(1.0 - np.random.uniform()) / rate

# Function to generate service time
def generate_service_time(category):
    return np.random.exponential(SERVICE_TIMES[category])

# Simulation function
def simulate_er_queue(arrival_rate):
    time = 0
    queue = {RED: [], YELLOW: [], GREEN: []}
    current_patient = None
    events = []  # to store event logs

    while time < SIMULATION_TIME:
        if current_patient is None or (queue[RED] and current_patient['category'] != RED):
            # If there is a red code patient waiting or no current patient, serve red
            if queue[RED]:
                current_patient = queue[RED].pop(0)
                current_patient['service_start'] = time
            else:
                # If no patient, generate next arrival
                next_arrival = generate_next_arrival(arrival_rate)
                category = np.random.choice(
                    [RED, YELLOW, GREEN], 
                    p=[ARRIVAL_RATES[RED], ARRIVAL_RATES[YELLOW], ARRIVAL_RATES[GREEN]]
                )
                queue[category].append({
                    'arrival_time': time + next_arrival, 
                    'service_time': generate_service_time(category), 
                    'category': category
                })
                events.append((time, 'arrival', category))
                time += next_arrival
                continue
        elif current_patient and time - current_patient['service_start'] >= current_patient['service_time']:
            # Current patient's service is done
            events.append((time, 'departure', current_patient['category']))
            current_patient = None
            continue

        # Check if service is interrupted by a red code patient
        for category in [YELLOW, GREEN]:
            if current_patient and current_patient['category'] == category and queue[RED]:
                # Service interrupted by red code arrival
                current_patient['remaining_service_time'] = current_patient['service_time'] - (time - current_patient['service_start'])
                queue[category].insert(0, current_patient)  # Put the patient back in the queue
                events.append((time, 'interruption', current_patient['category']))
                current_patient = None
                break

        # Increment time
        time += 1

    return events

# Run the simulation with different arrival rates
low_arrival_rate_events = simulate_er_queue(1/10)  # on average one patient every 10 minutes
high_arrival_rate_events = simulate_er_queue(1/5)  # on average one patient every 5 minutes

# Function to summarize events
def summarize_events(events):
    counts = {RED: 0, YELLOW: 0, GREEN: 0}
    for event in events:
        if event[1] == 'arrival':
            counts[event[2]] += 1
    print("Summary of events:")
    for category, count in counts.items():
        print(f"{category.capitalize()} code patients: {count}")

summarize_events(low_arrival_rate_events)
summarize_events(high_arrival_rate_events)