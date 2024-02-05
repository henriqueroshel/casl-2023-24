import numpy as np

def main():
    # define lower and upper bounds and length of vector
    A,B = 0,500
    N = 100
    # generate the vector
    random_values = np.random.uniform(0,500,100)

    # calculate the mean
    M = random_values.mean()
    # calculate standard deviation
    V = random_values.std(ddof=1)

    print(f'Average: {M:9f}')
    print(f'Standard deviation: {V:9f}')


if __name__ == '__main__':
    main()

    