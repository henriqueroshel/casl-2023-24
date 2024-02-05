# Let X be the output of stochastic process (.e.g, the estimated average in an experiment).

# Let us assume that X is uniformly distributed between 0 and 10.

# We wish to study the effect on the accuracy of the estimation in function of the number of experiments and in function of the confidence level.
# 1. Define properly all the input parameters
# 2. Write all the adopted formulas
# 3. Explain which python function you use to compute average, standard deviation and confidence interval

# 4. Plot the confidence interval and the accuracy 
# plot confidence interval with respect of number of samples
# plot accuracy with respect of number of samples and confidence level
# 5. Discuss the main conclusions drawn from the graphs.

# import here
import numpy as np
from scipy.stats import t,norm
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
# to avoid "jumps" in the plots
np.random.seed(42)


# choose input parameters confidence level
# these parameters are better set with parser
parser = argparse.ArgumentParser(description=                               'Generate iid samples.')
parser.add_argument('--low', type=float, default=0, help=                   'Lower bound of the distribution.')
parser.add_argument('--high', type=float, default=10, help=                 'Upper bound of the distribution.')
#just trying to predict the future requests, not really used in the code
# parser.add_argument('--samples', type=int, default=100, help=               'Number of samples to generate.')
parser.add_argument('--confidence_level', type=float, default=0.02, help=   'Number of simulations to run.') #not used, if you want to calculate another CI just add to append it
parser.add_argument('--distribution', type=str, default="Uniform", help=   'Number of simulations to run.')


# adopted formulas
# less than 30 samples i must use student t distribution
# more than 30 samples i should use normal z distribution
def conf_interval_calculator(x, conf_interval, df, empirical_mean, standard_error):
    samples=len(x)
    if (samples < 30):
        confidence_interval = t.interval(
            conf_interval,  # confidence level
            df,             # degrees of freedom
            empirical_mean, # sample mean
            standard_error  # standard error            
        )
    else:
        # z-distribution
        alpha = 1 - conf_interval
        z_value = -norm.ppf(alpha / 2)  # z-value
        margin_of_error = z_value * standard_error
        confidence_interval = (
            empirical_mean - margin_of_error, 
            empirical_mean + margin_of_error
        )

    return confidence_interval


# i extend the samples that i have to avoid having a segmented graph
def samples_constructor(x_values, num_samples, low, high, distribution):
    if x_values is None:  # If x_values is empty
        if distribution == "Uniform":
            x = np.random.uniform(low, high, num_samples)
            return x
        else:
            print("Not allowed")
            exit()
    else:  # If x_values is not empty
        num_to_generate = num_samples - len(x_values)
        if num_to_generate > 0:
            if distribution == "Uniform":
                additional_samples = np.random.uniform(low, high, num_to_generate)
                return np.concatenate([x_values, additional_samples])
            else:
                print("Not allowed")
                exit()
        else:
            return x_values[:num_samples]  # If num_samples is less than the length of x_values, then return a slice of x_values, shouldn't be the case or should never happen

# body of simulation, please note that at every run all the previous samples are "lost".
# this is not a bug, this is a feature :)

def run_simulator(x_values, num_samples, confidence_level, low, high, distribution):
    
    x = samples_constructor(
        x_values = x_values, 
        num_samples=num_samples, 
        low=low, 
        high=high, 
        distribution=distribution 
    )
    conf_interval = 1-confidence_level
    df = len(x) - 1 # degree of freedom for t's student distribution
    # print(df)

    empirical_mean = np.mean(x)
    empirical_std = np.std(x, ddof=1)  # using ddof=1 for sample standard deviation because lab0 taught me by getting low mark
    standard_error = empirical_std / np.sqrt(len(x))

    # print(f"Mean is {empirical_mean}")
    # print(f"empirical_std is {empirical_std}")
    # print(f"standard_error is {standard_error}")

    confidence_interval = conf_interval_calculator( x, conf_interval, df, empirical_mean, standard_error)

    # Empirical error is sup(CI)-inf(CI) / 2 
    # on wikipedia it is called standard error! https://en.wikipedia.org/wiki/Standard_error
    delta = (confidence_interval[1] - confidence_interval[0]) / 2
    relative_error = delta / empirical_mean
    # print(f"Relative Error is {relative_error}")


    accuracy = 1 - relative_error
    # print(f"Accuracy is {accuracy}")

    # appending parametrs to be used for graph purpose
    # conf_intervals_list.append(confidence_interval)
    # accuracy_list.append(accuracy)
    return confidence_interval, accuracy, empirical_mean, x


def lord_of_the_plots():
    fig, axes = plt.subplots(len(conf_intervals), 2, figsize=(15, 5*len(conf_intervals)))

    for ci_index, ci in enumerate(conf_intervals):
        # if only one confidence interval => axes[ci_index] will be 1D. 
        # handled separately, not sure if it exists another way
        if len(conf_intervals) == 1:
            ax1 = axes[0]
            ax2 = axes[1]
        else:
            ax1 = axes[ci_index][0]
            ax2 = axes[ci_index][1]

        # CI on the left
        upper_bounds = [conf_intervals_list[sample_index][ci_index][1] for sample_index in range(len(samples))]
        lower_bounds = [conf_intervals_list[sample_index][ci_index][0] for sample_index in range(len(samples))]
        ax1.plot(samples, upper_bounds, 'r-', label=f'Upper Bound')
        ax1.plot(samples, lower_bounds, 'b-', label=f'Lower Bound')
        ax1.scatter(samples, upper_bounds, color='r', s=10)  # show points
        ax1.scatter(samples, lower_bounds, color='b', s=10)  # show points
        ax1.set_xlabel('Sample Size, after sample size 100 the gap between sample sizes is 20')
        ax1.set_ylabel('Confidence Interval on interval width')
        ax1.legend()
        ax1.set_title(f"Confidence Level: {(1-ci)*100}%")

        # accuracies on the right
        accuracies = [accuracy_list[sample_index][ci_index] for sample_index in range(len(samples))]
        ax2.plot(samples, accuracies, 'g-', label=f'Accuracy')
        ax2.scatter(samples, accuracies, color='g', s=10)  # show points
        ax2.set_xlabel('Sample Size, after sample size 100 the gap between sample sizes is 20')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    plt.subplots_adjust(hspace=1)  # space for the plots
    plt.tight_layout()
    plt.savefig("accuracies_CI_plot.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()

    # i should loop for samples
    # i should loop for Confidence levesl
    x = None # samples array, at the beginning is nothing so it will be created later by the function
    samples = [10, 15, 24, 37, 50] + [i for i in range(70, 301, 2)] # lazy writing, 70,90,110,130...to 1000
    conf_intervals = [0.1, 0.05, 0.02, 0.01] # 90% 95% 98% 99%

    # data structure to save accuracy and confidence intervals
    conf_intervals_list = [[] for _ in samples]
    accuracy_list = [[] for _ in samples]
    empirical_means = [[] for _ in samples]

    # for each sample in my sample list
    for sample_index, sample_size in enumerate(samples):
        #calculate, on this sample, the confidence interval levels that i have in my list
        for ci in conf_intervals:
            # rreturn conf intervals values, accuracy, emp mean and the array x used
            ci_v, acc, empmean, x = run_simulator(x_values = x, num_samples=sample_size, confidence_level = ci, low = args.low, high = args.high , distribution=args.distribution)
            # not trusting python to append in global in the function, so i used loval values here in the main
            conf_intervals_list[sample_index].append(ci_v)
            accuracy_list[sample_index].append(acc)
            empirical_means[sample_index].append(empmean)

    # plot with lord of the plots!
    lord_of_the_plots()
    # plotto beggins is not trust worthy
    # plotto_beggins()