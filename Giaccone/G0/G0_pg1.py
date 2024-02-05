import numpy as np
from numpy.random import uniform

def main() :
    # Limits for random numbers
    A = 0
    B = 500

    # Number of real values generated
    N = 100

    # Numbers generation
    rand_nums = uniform(A,B,N)

    mean = np.mean(rand_nums)
    sd = np.std(rand_nums)

    print("Mean :", mean, "- Standard deviation :", sd)

if __name__ == "__main__":
    main()