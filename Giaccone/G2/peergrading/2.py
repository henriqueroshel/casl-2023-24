import numpy as np
import math
from scipy.stats import t
import matplotlib.pyplot as plt

# Random seed setted for reproducibility reasons
np.random.seed(1)

# Definition of constants
MAX_TIME = 1000 # Maximum simulation time, corresponding to the final number of samples extracted
confidence_level = 0.98 # Fixed confidence level for the run with an increasing number of samples

#Definition of variables

lower_bounds = [] # List for storing all the lower bounds of the confidence intervals when the number of samples increases
upper_bounds = [] # List for storing all the upper bounds of the confidence intervals when the number of samples increases
accuracies = [] # List for storing all the accuracies when the number of samples increases
means = [] # List for storing the empirical mean when a new sample is extracted
samples = [np.random.uniform(0,10) for _ in range(10)] # List for storing all the iid samples in the first run. Initialized with 10 random samples
curr_time = 10 # Represents the current number of samples and it is used also for stopping the simulation when the MAX_TIME has been reached

'''
Main cicle for extracting iid random uniform samples (values between 0 and 10 )
and computing all the critical measures such as accuracy, empirical mean and confidence intervals 
First, a sample is extracted and inserted in the samples list, then the time is incremented 
(coinciding with the number of samples) and mean and standard deviation are computed from the list.
They will be used for computing the confidence interval through the t.interval() function of the scipy.stats module.
After the confidence interval has been obtained, the relative error and consequently the accuracy are computed.
At the end of every iteration, the accuracy, lower bound of the confidence interval and upper bound for the current samples are stored.
'''
while curr_time < MAX_TIME:
    samples.append(np.random.uniform(0, 10))
    curr_time += 1 
    mean = np.mean(samples)
    means.append(mean)
    stdev = np.std(samples, ddof=1) # ddof = 1 for computing the unbiased standard deviation
    size = stdev / math.sqrt(curr_time) # size is the coefficient that multiplies the t-score of the t-student distribution for the confidence interval
    
    # The parameters of t.interval() are the confidence level (0.98), 
    # the degrees of freedom (N_samples-1), the empirical mean and the multiplier of the t-score
    confidence_interval = t.interval(confidence_level, curr_time - 1, mean, size)
    delta = (confidence_interval[1]-confidence_interval[0])/2
    relative_error = delta/mean 
    accuracy = 1 - relative_error
    
    lower_bounds.append(confidence_interval[0])
    upper_bounds.append(confidence_interval[1])
    accuracies.append(accuracy)

# Experiments for the observation of accuracy and confidence intervals when 
# the number of samples is fixed and the confidence level increases

# Initialization of the confidence levels (90 levels between 0.1 and 0.99 included)
confidence_levels = np.linspace(0.1, 0.99, endpoint=True, num=90)

# The following lists are defined for storing all the results for lower bounds, 
# upper bounds and accuracy for the second experiment
lb = []
ub = []
accs = []

# The main cycle iterates over all the confidence levels and computes 
# the confidence interval using all the samples extracted previously
# Then, the accuracy is computed and all the results are stored.
for level in confidence_levels:
    confidence_interval = t.interval(level, curr_time - 1, mean, size)
    delta = (confidence_interval[1]-confidence_interval[0])/2
    relative_error = delta/mean 
    accuracy = 1 - relative_error
    lb.append(confidence_interval[0])
    ub.append(confidence_interval[1])
    accs.append(accuracy)


# Visualization of the four results 
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

# Visualization of the trend of the accuracy while the number of samples increases.
# The x axis represents the number of samples (from 10 to 1000) and the y axis the accuracy
# It can be noticed that the accuracy tends to increase with more samples. 
ax[0, 0].plot(range(10,MAX_TIME), accuracies)
ax[0, 0].set_xlabel("Number of Samples")
ax[0, 0].set_ylabel("Accuracy")
ax[0, 0].grid()

# Visualization of the trend of the confidence interval while the number of sample increases
# The x axis represents the number of samples (from 10 to 1000) and the y axis represents the value
# of the empirical mean in case of the black line and the range of the confidence interval
# in the case of the blue lines (filled to better visualize the interval).
# It can be noticed that the range of the interval tends to decrease with more samples. 
ax[0, 1].plot(range(10,MAX_TIME), means, color='black')
ax[0, 1].legend(["Empirical Mean"])
ax[0, 1].fill_between(range(10, MAX_TIME), upper_bounds, lower_bounds, color='blue', alpha=0.5)
ax[0, 1].set_xlabel("Number of Samples")
ax[0, 1].set_ylabel("Confidence Interval")
ax[0, 1].set_yticks(np.linspace(2.5, 7.5, 6, endpoint=True))
ax[0, 1].grid()

# Visualization of the trend of the accuracy while the confidence level increases and the number of samples is fixed to 1000.
# The x axis represents confidence level (from 0.1 to 0.99 included) and y axis the accuracy
# It can be noticed that the accuracy drops rapidly, especially when the confidence level is between 0.9 and 0.99
ax[1, 0].plot(confidence_levels, accs)
ax[1, 0].set_xlabel("Confidence Level")
ax[1, 0].set_ylabel("Accuracy")
ax[1, 0].grid()

# Visualization of the trend of the confidence interval while the confidence level increases and the number of samples is fixed to 1000.
# The x axis represents the confidence level (from 0.1 to 0.99) and the y axis represents the value
# of the empirical mean in case of the black line, and the range of the confidence interval
# in the case of the blue lines (filled to better visualize the interval).
# It can be noticed that the range of the interval tends to increase with an higher level of confidence, especially
# when the confidence level is between 0.9 and 0.99
ax[1, 1].plot(confidence_levels, [mean for _ in range(len(ub))], color='black')
ax[1, 1].legend(["Empirical Mean"])
ax[1, 1].fill_between(confidence_levels, ub, lb, color='blue', alpha=0.5)
ax[1, 1].set_xlabel("Confidence Level")
ax[1, 1].set_ylabel("Confidence Interval")
ax[1, 1].grid()

plt.show()