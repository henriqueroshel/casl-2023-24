import numpy as np

samples = np.random.uniform(0,500,100)
avg = samples.mean()
std = samples.std()

print("The average is:", avg)
print("The standard deviation is:", std)