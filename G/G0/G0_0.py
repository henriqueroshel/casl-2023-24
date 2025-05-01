from random import uniform
from numpy import mean, std

def main():
    # define lower and upper bounds and length of vector
    A,B = 0,500
    N = 100
   # generate the vector
    random_values = [ uniform(A,B) for _ in range(N) ]

    # calculate the mean
    M = sum(random_values) / N
    # calculate standard deviation
    V = ( sum([ (x-M)**2 for x in random_values ]) / (N-1) ) ** .5

    print(f'Average: \n{M:9f}\t ({mean(random_values):9f})')
    print(f'Standard deviation:\n{V:9f}\t({std(random_values, ddof=1):9f})')


if __name__ == '__main__':
    main()

    