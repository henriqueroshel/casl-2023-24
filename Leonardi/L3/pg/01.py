#------------------------------------------------------------------------------------------------#

# IMPORTS
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------#

# FUNCTIONS

# The argument size will be used for the different sample sizes

# GENERATION WITH INVERSE TRANSFORM TECHNIQUE

# Inverse transform for generating ranfom numbers from uniform [0, 1]
def generate_uniform(size):
    return np.random.rand(size)

# Inverse transform generation of the distributions
def generate_rayleigh_inverse(sigma, size):
    u = generate_uniform(size)
    return sigma * np.sqrt(-2 * np.log(1-u))

def generate_lognormal_inverse(mu, sigma, size):
    u = generate_uniform(size)
    return np.exp(mu + sigma * np.sqrt(2) * stats.norm.ppf(u))

def generate_beta_inverse(alpha, beta, size):
    u = generate_uniform(size)
    return np.power(u, 1/alpha) * np.power(1-u, 1/beta)

def generate_chi_square_inverse(n, size):
    u = generate_uniform(size)
    chi_square_data = np.zeros(size)
    for i in range(size):
        exponential_sum = 0
        for j in range(n):
            exponential_sum += -np.log(1 - generate_uniform(1))
        chi_square_data[i] = exponential_sum 
    return chi_square_data

def generate_rice_inverse(nu, sigma, size):
    u = generate_uniform(size)
    chi_square_part = generate_chi_square_inverse(2 * nu, size)
    normal_part = np.random.normal(0, sigma, size)
    return np.sqrt(chi_square_part + normal_part ** 2)

# GENERATION WITH NUMPY BUILT-IN FUNCTIONS

# Numpy generation of the distributions
def generate_rayleigh(sigma, size):
    return np.random.rayleigh(sigma, size)

def generate_lognormal(mu, sigma, size):
    return np.random.lognormal(mu, sigma, size)

def generate_beta(alpha, beta, size):
    return np.random.beta(alpha, beta, size)

def generate_chi_square(n, size):
    return np.random.chisquare(n, size)

def generate_rice(nu, sigma, size):
    return np.sqrt(np.random.chisquare(nu, size) + np.random.normal(0, sigma, size) ** 2)

# Empirical moments
def empirical_moments(data):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data, ddof=1)                                      # calculate the sample variance (with Bessel's correction)
    return mean, var

#------------------------------------------------------------------------------------------------#

# PARAMETERS FOR THE DISTRIBUTIONS

# Rayleigh distribution parameters
sigma_rayleigh = 1.0
# Lognormal distribution parameters
mu_lognormal = 1.0
sigma_lognormal = 0.5
# Beta distribution parameters
alpha_beta = 2.0
beta_beta = 4.0
# Chi square distribution parameters
n_chi_square = 3
# Rice distribution parameters
nu_rice = 1
sigma_rice = 0.7

sample_sizes = [100, 10000, 100000]

#------------------------------------------------------------------------------------------------#

# GENERATION

for size in sample_sizes:
    
    # Generate random variables with numpy
    # rayleigh_data = generate_rayleigh(sigma_rayleigh, size)
    # lognormal_data = generate_lognormal(mu_lognormal, sigma_lognormal, size)
    # beta_data = generate_beta(alpha_beta, beta_beta, size)
    # chi_square_data = generate_chi_square(n_chi_square, size)
    # rice_data = generate_rice(nu_rice, sigma_rice, size)

    # Generate random variables with inverse transform 
    rayleigh_data = generate_rayleigh_inverse(sigma_rayleigh, size)
    lognormal_data = generate_lognormal_inverse(mu_lognormal, sigma_lognormal, size)
    beta_data = generate_beta_inverse(alpha_beta, beta_beta, size)
    chi_square_data = generate_chi_square_inverse(n_chi_square, size)
    rice_data = generate_rice_inverse(nu_rice, sigma_rice, size) 
    
    # Analytical moments
    analytical_moments = {
        'Rayleigh': [sigma_rayleigh * np.sqrt(np.pi / 2), (4 - np.pi) * sigma_rayleigh ** 2 / 2],
        'Lognormal': [np.exp(mu_lognormal + sigma_lognormal ** 2 / 2), (np.exp(sigma_lognormal ** 2) - 1) * np.exp(2 * mu_lognormal + sigma_lognormal ** 2)],
        'Beta': [alpha_beta / (alpha_beta + beta_beta), (alpha_beta * beta_beta) / ((alpha_beta + beta_beta) ** 2 * (alpha_beta + beta_beta + 1))],
        'Chi-Square': [n_chi_square, 2 * n_chi_square],
        'Rice': [(np.sqrt(np.pi) / 2) * np.exp(-(nu_rice ** 2) / (2 * sigma_rice ** 2)), (1 + nu_rice ** 2 / sigma_rice ** 2) - (np.pi / 2)],
    }
    
    # Empirical moments
    moments = {
        'Rayleigh': empirical_moments(rayleigh_data),
        'Lognormal': empirical_moments(lognormal_data),
        'Beta': empirical_moments(beta_data),
        'Chi-Square': empirical_moments(chi_square_data),
        'Rice': empirical_moments(rice_data),
    }
    
    print(f"Sample size: {size}")
    for distribution, (mean_empirical, var_empirical) in moments.items():
        mean_analytical, var_analytical = analytical_moments[distribution]
        print(f"{distribution} - Empirical Mean: {mean_empirical:.4f}, Analytical Mean: {mean_analytical:.4f}")
        print(f"{distribution} - Empirical Variance: {var_empirical:.4f}, Analytical Variance: {var_analytical:.4f}")
        
    
    # Compare CDFs for Rayleigh distribution
    plt.figure()
    plt.hist(rayleigh_data, bins=50, density=True, alpha=0.5, label='Empirical PDF')
    x = np.linspace(0, 10, 1000)
    plt.plot(x, stats.rayleigh.pdf(x, sigma_rayleigh), 'r', lw=2, label='Analytical PDF')
    plt.title(f'Rayleigh Distribution (n={size})')
    plt.legend(loc='upper right')
    plt.show()

    # # Create a single figure for all three Rayleigh distribution plots
    # plt.figure(figsize=(12, 4))

    # for i, size in enumerate(sample_sizes):
    #     plt.subplot(1, 3, i+1)  # Create subplots within the same figure
    #     rayleigh_data = generate_rayleigh_inverse(sigma_rayleigh, size)
    #     plt.hist(rayleigh_data, bins=50, density=True, alpha=0.5, label=f'n={size}')
    #     x = np.linspace(0, 10, 1000)
    #     plt.plot(x, stats.rayleigh.pdf(x, sigma_rayleigh), 'r', lw=2, label='Analytical PDF')
    #     plt.title(f'Rayleigh Distribution (n={size})')
    #     plt.legend(loc='upper right')

    # plt.tight_layout()  # Adjust subplot spacing for a clean layout
    # plt.show()