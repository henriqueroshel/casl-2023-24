import numpy as np
from scipy.stats import uniform

import numpy as np


# Define the parameters
N = 100
A = 0
B = 500

                # Generate the random values in scipy
random_values = uniform.rvs(loc=A, scale=B-A, size=N)
integer_random_values = [int(x) for x in random_values]

# Print the generated values
print(integer_random_values)

                 # Generate the random values in numpy
random_numbers = np.random.uniform(A, B, N)
integer_random_numbers = [round(x) for x in random_numbers]

# Print the generated values
print(integer_random_numbers)