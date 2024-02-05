import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define the Rayleigh random variable generator function
def rayleigh_rv(sigma, size=1):
    """Generate Rayleigh distributed random variables."""
    return sigma * np.sqrt(-2 * np.log(np.random.uniform(0, 1, size)))

# Set the parameters for the Rayleigh distribution
sigma = 2  # Scale parameter for the Rayleigh distribution

# Sizes of sample extractions to test
sample_sizes = [1000, 10000, 100000]

# Preparing a dictionary to store the empirical and analytical CDFs
cdf_comparison = {}

# Generate random variables and calculate empirical CDF for each sample size
for size in sample_sizes:
    samples = rayleigh_rv(sigma, size)

    # Calculate empirical CDF
    count, bins_count = np.histogram(samples, bins=1000)
    pdf = count / size
    cdf = np.cumsum(pdf)

    # Calculate analytical CDF using scipy's rayleigh distribution
    analytical_cdf = stats.rayleigh.cdf(bins_count[1:], loc=0, scale=sigma)

    # Store the CDFs for comparison
    cdf_comparison[size] = (bins_count[1:], cdf, analytical_cdf)

# Select the largest sample size for plotting
bins, empirical_cdf, analytical_cdf = cdf_comparison[100000]

# Plot the empirical CDF and the analytical CDF
plt.figure(figsize=(10, 6))
plt.plot(bins, empirical_cdf, label="Empirical CDF", color='blue')
plt.plot(bins, analytical_cdf, label="Analytical CDF", color='red', linestyle='--')
plt.title("Comparison between Empirical and Analytical CDF (Rayleigh Distribution)")
plt.xlabel("Value")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)

# Save the plot to a file
image_file_path = 'empirical_analytical_cdf_rayleigh.png'
plt.savefig(image_file_path)
plt.show()