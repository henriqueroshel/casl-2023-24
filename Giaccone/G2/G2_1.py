import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt

MIN_, MAX_ = 0, 10
CONFIDENCE_LEVELS = np.arange(0.9, 1, 0.001)
N_EXPERIMENTS = np.arange(10,1001,10)

# auxiliary arrays for plots
CONFIDENCE_LEVELS_XTICKS = np.arange(90, 101, 2)
N_EXPERIMENTS_XTICKS = np.arange(0,1001,200), np.arange(0,1001,50)


def accuracy(ci, avg):
    ci_lower, ci_upper = ci
    eps = (ci_upper-ci_lower) / (2*avg) # relative error
    acc = 1 - eps # accuracy
    return acc

def simulation(X, confidence):
    avg = X.mean()
    std = X.std(ddof=1)
    n = len(X)

    '''
    # Alternative computation of confidence interval
    alpha = 1-confidence
    semi_width = t.ppf(1-alpha/2, n-1) * std/(n**.5)
    ci_lower = avg - semi_width
    ci_upper = avg + semi_width
    ci = ci_lower,ci_upper
    '''
    
    # Compute confidence interval and accuracy
    ci = t.interval(confidence, n-1, avg, std)
    acc = accuracy(ci, avg)
    return ci, acc


def main():
    print('------------------------------------')
    print("SIMULATION FOR CONFIDENCE LEVEL:")
    n = int(input("Number of experiments: "))

    # Generation of n experiments uniformly distributed
    X = np.random.uniform(MIN_, MAX_, N_EXPERIMENTS[0])

    # loop for the considered confidence levels values
    ci_lower= np.zeros(len(CONFIDENCE_LEVELS))
    ci_upper = np.zeros(len(CONFIDENCE_LEVELS))
    accs = np.zeros(len(CONFIDENCE_LEVELS))
    for i, confidence in enumerate(CONFIDENCE_LEVELS):
        ci, acc = simulation(X, confidence)
        ci_lower[i] = ci[0]
        ci_upper[i] = ci[1]
        accs[i] = acc

    # plots
    fig, (ax0,ax1) = plt.subplots(2, sharex=True)
    fig.suptitle(f"Number of experiments: {n}")
    ax0.set_ylabel("Accuracy (%)")
    ax1.set_ylabel("Confidence interval")
    ax1.set_xlabel("Confidence level (%)")
    ax0.set_xticks(np.arange(90, 101, 2), minor=False)
    ax0.set_xticks(np.arange(90, 101, .5), minor=True)
    
    confidences_pct = CONFIDENCE_LEVELS*100
    ax0.plot(confidences_pct, accs, '-')
    ax1.plot(confidences_pct, ci_lower, '.-', label='Lower bound')
    ax1.plot(confidences_pct, ci_upper, '.-', label='Upper bound')
    ax0.grid()
    ax1.grid()
    plt.show()
    print("-------------------------------------")
    
    print("-------------------------------------")
    print("SIMULATION FOR NUMBER OF EXPERIMENTS:")
    confidence = int(input('Confidence level (%): ')) / 100

    # loop for the considered number of experiments
    ci_lower= np.zeros(len(N_EXPERIMENTS))
    ci_upper = np.zeros(len(N_EXPERIMENTS))
    accs = np.zeros(len(N_EXPERIMENTS))

    # Generation of n experiments uniformly distributed
    X = np.random.uniform(MIN_, MAX_, N_EXPERIMENTS[0])
    for i, n in enumerate(N_EXPERIMENTS):
        ci, acc = simulation(X, confidence)
        ci_lower[i] = ci[0]
        ci_upper[i] = ci[1]
        accs[i] = acc

        # Generate next values
        if len(N_EXPERIMENTS) > i+1:
            next_n = N_EXPERIMENTS[i+1]
            next_X = np.random.uniform(MIN_, MAX_, next_n-n)
            X = np.concatenate([X,next_X])



    # plots
    fig, (ax0,ax1) = plt.subplots(2, sharex=True)
    fig.suptitle(f"Confidence level: {confidence:.0%}")
    ax0.set_ylabel("Accuracy (%)")
    ax1.set_ylabel("Confidence interval")
    ax1.set_xlabel("Number of simulations")
    ax0.set_xticks(N_EXPERIMENTS_XTICKS[0], minor=False)
    ax0.set_xticks(N_EXPERIMENTS_XTICKS[1], minor=True)

    ax0.plot(N_EXPERIMENTS, accs, '.-')
    ax1.plot(N_EXPERIMENTS, ci_lower, '.-')
    ax1.plot(N_EXPERIMENTS, ci_upper, '.-')
    ax0.grid()
    ax1.grid()
    plt.show()
    print("-------------------------------------")

    print("-------------------------------------")
    print("SIMULATION FOR A SINGLE PAIR CONFIDENCE AND NUMBER OF EXPERIMENTS:")
    confidence = int(input('Confidence level (%): ')) / 100
    n = int(input("Number of experiments: "))
    X = np.random.uniform(MIN_, MAX_, n)
    ci, acc = simulation(X, confidence)
    print(f'Sample average: {sum(ci)/2:.3f}')
    print(f'Confidence interval: [{ci[0]:.3f}, {ci[1]:.3f}]')
    print(f'Accuracy: {acc:.3f}')
    print("-------------------------------------\n")

if __name__ == '__main__':
    main()