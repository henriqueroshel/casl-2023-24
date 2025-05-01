import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from math import log, cos, sin, pi, exp, factorial, gamma
from scipy import stats

def poisson(lamb):
    # generate a value from poisson random variable
    # computes p(X=k) for X \sim Poisson(lambda_)
    pdf = lambda k : (lamb**k)*exp(-lamb)/factorial(k)
    u = uniform(0,1)
    k = 0
    cumulative_prob = pdf(k)
    while cumulative_prob < u:
        k += 1
        cumulative_prob += pdf(k)
    return k

def normal(mu=0, sigma=1):
    n = 12
    sum_u = sum((uniform(0,1) for _ in range(n)))
    x = (sum_u-n/2)
    x = mu + sigma*x
    return x

def rayleigh(sigma):
    # inverse transform technique
    u = uniform(0,1)
    x = sigma * (-2*log(1-u)) ** .5
    return x

def lognormal(mu, sigma):
    # apply the exponential to the normal rv
    x = normal()
    y = exp(mu + sigma*x)
    return y

def beta(a, b, c=None):
    # Acceptance/Rejection Technique
    Beta_ab = (gamma(a)*gamma(b))/gamma(a+b)
    if not c:
        # get max value of pdf if not given
        step=0.0001
        x = np.arange(0,1+step,step)
        beta_pdf = ( x**(a-1)*(1-x)**(b-1) ) / Beta_ab
        c = beta_pdf.max()
    while True:
        x = uniform(0, 1)
        y = uniform(0, c)
        f_x = ( x**(a-1)*(1-x)**(b-1) ) / Beta_ab
        if y<=f_x:
            return x

def chi2(n):
    # convolution method of n standard normal variates
    z = ( normal()**2 for _ in range(n) )
    x = sum(z)
    return x

def rice(v,sigma):
    # generate a value from a rice random variable based on poisson and chi2
    P = poisson(.5*(v/sigma)**2)
    X = chi2(2*P + 2)
    R = sigma * X**.5
    return R

def generate(n, dist_generator, parameters_dict):
    # generate n random instances of the given distribution
    values = np.zeros(n)
    for i in range(n):
        values[i] = dist_generator(**parameters_dict)
    return values

def confidence_interval(values, confidence=.98):
    # computes the confidence interval for the average
    # of a particular measure where values is a list
    n = len(values)                 # number samples
    avg = np.mean(values)           # sample mean
    std = np.std(values, ddof=1)    # sample standard deviation
    ci_low, ci_upp = stats.t.interval(confidence, n-1, avg, std/n**.5)
    delta = (ci_upp-ci_low)/2
    return avg, delta
def accuracy(value, delta):
    # computes the accuracy of a measure given its value
    # and the semi-width (delta) of the confidence interval    
    eps = delta / value # relative error
    acc = 1 - eps # accuracy
    return max(acc, 0) # return only non-negative values

def plot_dist(empirical_values, pdf_function):
    # compare pdf with the empirical frequency obtained
    x = np.linspace(empirical_values.min(), empirical_values.max(), 1000)
    pdf = pdf_function(x)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    n=len(empirical_values)
    histogram = ax1.hist( empirical_values, bins=np.linspace(min(empirical_values),max(empirical_values),100), 
                          alpha=0.65, color='tab:blue')
    analytic = ax2.plot(x, pdf, color='tab:red')
    
    ax1.set_xlabel('value')
    ax1.set_ylabel('Histogram counts', color='tab:blue')
    ax2.set_ylabel('pdf f(x)', color='tab:red')
    ax1.set_ylim(bottom=0, top=max(histogram[0])*1.05)    
    ax2.set_ylim(bottom=0, top=max(pdf)*1.05)
    ax1.legend([histogram[2], analytic[0]], ['Empirical values', 'Analytical pdf' ])
    plt.show()


if __name__=='__main__':
    N = [100,10000,100000]
    conf_level = .95

    # Beta distribution parameters
    a,b = 1.5, 4.5

    # get max value of beta pdf (i.e. c) in acceptance-rejection technique
    Beta_ab = (gamma(a)*gamma(b))/gamma(a+b) # beta function B(a,b)
    step=0.00001
    x = np.arange(0,1+step,step)
    beta_pdf = ( x**(a-1)*(1-x)**(b-1) ) / Beta_ab
    c = beta_pdf.max()    

    # analytical moments
    analytical_mean, analytical_var = stats.beta.stats(a, b, moments='mv')

    print(f'X ~ Beta(\u03b1={a}, \u03b2={b})')
    print('--- Analytical moments ---')
    print(f'- First moment: \u03bc={analytical_mean:.6f}')
    print(f'- Second moment: \u03c3\u00b2={analytical_var:.6f}')
    for n in N:
        print(f'\n---   n={n}   ---')
        beta_values = generate(n, beta, {'a':a,'b':b})
        mean, mean_delta = confidence_interval(beta_values)
        var, var_delta = confidence_interval((beta_values-mean)**2)
        print(f'- First moment: \u03bc={mean:.5f} \u00b1 {mean_delta:.5f} - accuracy: {accuracy(mean, mean_delta):.3f}')
        print(f'- Second moment: \u03c3\u00b2={var:.6f} \u00b1 {var_delta:.5f} - accuracy: {accuracy(var, var_delta):.3f}')

        plot_dist(beta_values, pdf_function=lambda x:(x**(a-1)*(1-x)**(b-1))/Beta_ab)