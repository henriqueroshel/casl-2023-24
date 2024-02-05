import random


class PriorityStatus:
    def __init__(self):
        self.interrupted_at = None  # Time at which the service was interrupted
        self.was_interrupted = False  # Flag to check if the service was interrupted


class Customer:
    def __init__(self, color_code, service_time, arrival_time):
        self.color_code = color_code  # 0 RED, 1 YELLOW, 2 GREEN
        self.service_time = service_time  # Time required to serve the customer
        self.original_service_time = service_time  # Store original service time for statistics
        self.priority_status = PriorityStatus()
        self.arrival_time = arrival_time  # Time at which the customer arrives


class Team:
    def __init__(self):
        self.current_customer = None  # The customer currently being served by the team


class Statistics:
    def __init__(self):
        self.total_customers = 0
        self.customers_served = 0
        self.total_waiting_time = 0
        self.total_service_time = 0
        self.interrupted_services = 0

    def print_stats(self):
        avg_waiting_time = self.total_waiting_time / self.total_customers if self.total_customers != 0 else 0
        avg_service_time = self.total_service_time / self.customers_served if self.customers_served != 0 else 0
        print("Simulation Statistics:")
        print("Total customers arrived:", self.total_customers)
        print("Total customers served:", self.customers_served)
        print("Total interrupted services:", self.interrupted_services)
        print("Average waiting time: {:.2f} hours".format(avg_waiting_time))
        print("Average service time: {:.2f} hours".format(avg_service_time))


def generate_service_time(mean, std_dev):
    # Generate service time based on the provided mean and standard deviation
    return max(random.normalvariate(mean, std_dev), 0.1)


def cust_can_arrive(customer_queue, time, team, stats, arrival_rate, mean_service_time, std_service_time):
    # Simulate customer arrival based on the provided arrival rate
    if random.random() < arrival_rate:
        color_code = random.choices([0, 1, 2], weights=[1 / 6, 1 / 3, 1 / 2])[0]
        service_time = generate_service_time(mean_service_time, std_service_time)
        new_customer = Customer(color_code, service_time, time)  # Pass the current time as the arrival time
        customer_queue.append(new_customer)  # Add the customer at the end of the queue
        stats.total_customers += 1

        # Check for priority and interrupt service if necessary
        if team.current_customer and team.current_customer.color_code > color_code:  # If this color code has more urgency than the one processed
            interrupted_customer = team.current_customer
            # Store in the interrupted customer some utility variables
            interrupted_customer.priority_status.interrupted_at = time
            interrupted_customer.priority_status.was_interrupted = True
            team.current_customer = new_customer
            # Move the interrupted customer at the start of the queue to be processed as soon as the higher priority one finishes
            customer_queue.insert(0, interrupted_customer)
            stats.interrupted_services += 1


def pass_time(team, customer_queue, time, stats):
    # If there's a current customer being served
    if team.current_customer:
        team.current_customer.service_time -= 1  # Decrement service time
        if team.current_customer.service_time <= 0:
            stats.customers_served += 1
            # Add the original service time of the customer to the total service time
            stats.total_service_time += team.current_customer.original_service_time
            team.current_customer = None

        # If there's no current customer being served and the queue is not empty, serve the next customer
    if not team.current_customer and len(customer_queue) > 0:
        # Remove the next customer from the queue (FIFO)
        next_customer = customer_queue.pop(0)

        # Calculate waiting time based on whether the customer was interrupted earlier or not
        if not next_customer.priority_status.was_interrupted:
            waiting_time = time - next_customer.arrival_time  # Calculate waiting time for non-interrupted customer
        else:
            # Calculate waiting time from the moment the customer was interrupted
            waiting_time = time - next_customer.priority_status.interrupted_at
        stats.total_waiting_time += waiting_time
        team.current_customer = next_customer


if __name__ == "__main__":
    random.seed(309345)
    max_sim_time = 10000  # Total simulation time
    service_time_normal_ranges = [(3, 1), (4, 2), (5, 3)]  # Tuples of (mean, standard deviation)
    # Different inter-arrival rates, basically the probability of a patient arriving
    inter_arrival_times = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    results = []  # List to store results

    # Run simulations
    for mean_service_time, std_service_time in service_time_normal_ranges:  # For x For each variable of the similation
        for arrival_rate in inter_arrival_times:
            team = Team()
            # List instead of queue, to better manage substituted customers with things like: list.insert(0, interrupted_customer)
            customer_queue = list()
            stats = Statistics()
            # Create a first customer-patient for initialization
            starting_customer = Customer(0, generate_service_time(mean_service_time, std_service_time), 0)
            customer_queue.append(starting_customer)

            for time in range(max_sim_time):  # Run the simulation until max time
                cust_can_arrive(customer_queue, time, team, stats, arrival_rate, mean_service_time, std_service_time)
                pass_time(team, customer_queue, time, stats)

            # Store the results in a dictionary
            simulation_result = {
                "mean_service_time": mean_service_time,
                "std_service_time": std_service_time,
                "arrival_rate": arrival_rate,
                "total_customers": stats.total_customers,
                "customers_served": stats.customers_served,
                "interrupted_services": stats.interrupted_services,
                "average_waiting_time": stats.total_waiting_time / stats.total_customers,
                "average_service_time": stats.total_service_time / stats.customers_served,
            }

            # Append the results to the list
            results.append(simulation_result)

    header = ["|Service Time (Mean, Std)|",
              "Arrival Rate|",
              "Total Customers|",
              "Customers Served|",
              "Interrupted Services|",
              "Avg Waiting Time|",
              "Avg Service Time|"]
    row_format = "{:<20}" + "{:<15}" * (len(header) - 1)  # Adjusted for a more compact layout

    # Print the results in a table-like format
    print("-" * len(row_format.format(*header)))
    print(row_format.format(*header))
    print("-" * len(row_format.format(*header)))

    for result in results:
        this_service_time = f"{result['mean_service_time']}, {result['std_service_time']}"
        row = [this_service_time,
               result["arrival_rate"],
               result["total_customers"],
               result["customers_served"],
               result["interrupted_services"],
               f'{result["average_waiting_time"]:.2f}',
               f'{result["average_service_time"]:.2f}']

        print(row_format.format(*row))

    print("-" * len(row_format.format(*header)))
