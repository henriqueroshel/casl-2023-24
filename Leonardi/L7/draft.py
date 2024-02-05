import numpy as np
import matplotlib.pyplot as plt

def mean_field_sir(N, beta, gamma, days, rho_func):
    # Initial conditions
    S = N - 1
    I = 1
    R = 0

    # Simulation parameters
    dt = 1  # Time step (days)

    # Arrays to store results
    S_values = np.zeros(days+1)
    S_values[0] = S
    I_values = np.zeros(days+1)
    I_values[0] = I
    R_values = np.zeros(days+1)
    R_values[0] = R

    # Simulation loop
    for day in range(days):
        # Mobility restriction factor
        rho = rho_func(day)

        # SIR model equations
        dS = -rho * beta * S * I / N * dt
        dI = (rho * beta * S * I / N - gamma * I) * dt
        dR = gamma * I * dt

        # Update variables
        S += dS
        I += dI
        R += dR

        # Append results to arrays
        S_values[day+1] = S
        I_values[day+1] = I
        R_values[day+1] = R

    return S_values, I_values, R_values

def mobility_restrictions(day):
    # Example of a basic strategy with increased restrictions after 30 days
    if day < 31:
        return 1.0  # No restrictions
    else:
        return 0.5  # 50% reduction in mobility

# Simulation parameters
population_size = 50000000  # 50 million
initial_infected = 1
infection_rate = 4.0 / population_size
recovery_rate = 1 / 14
simulation_days = 180  # 6 months

# Run simulation
S, I, R = mean_field_sir(population_size, infection_rate, recovery_rate, simulation_days, mobility_restrictions)


print(S.max(),S.min(),I.max(),I.min(),R.max(),R.min(), sep='\n')

# Plotting the results
days = np.arange(simulation_days + 1)
# plt.plot(days, S, label='Susceptible')
# plt.plot(days, I, label='Infected')
# plt.plot(days, R, label='Recovered')
# plt.xlabel('Days')
# plt.ylabel('Population')
# plt.title('Mean-Field SIR Model with Mobility Restrictions')
# plt.legend()
# plt.show()
