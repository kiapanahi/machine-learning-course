import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data():
    data = pd.read_csv('./src/country_stat.csv')
    return data


def gd_fit(X, y, alpha, max_iterations):
    [m, n] = X.shape
    X2 = np.ones([m, n+1])
    X2[:, 1:n+1] = X[:, 0:n]
    theta = np.random.uniform(0, 1, [n+1, 1])

    E = np.zeros([1, max_iterations])

    for iter in range(max_iterations):

        actual_y = X2@theta
        e = (y-actual_y)

        E[0, iter] = (np.transpose(e)@e)/m

        grad = np.zeros([n+1, 1])

        for i in range(m):
            for j in range(n+1):
                grad[j, 0] = grad[j, 0] - (X2[i, j]*e[i, 0])

        theta = theta - alpha*grad
        alpha = 0.99*alpha

    T = np.arange(0, max_iterations, 1)
    plt.plot(T, E[0, :])
    plt.show()


if __name__ == "__main__":
    data = load_data()
    gdp_per_capita = data['GDP per capita']
    life_satisfaction = data['Life satisfaction']

    gd_fit(np.c_[gdp_per_capita], np.c_[life_satisfaction],
           0.000000000000000000001, 100)


# self written
##############################################################

def fitness(arr, theta):
    total_penalty = 0
    for actual in arr:
        x_penalty = penalty(actual, h(theta, actual))
        total_penalty += x_penalty
    return total_penalty/2


def penalty(calculated, actual):
    return (actual - calculated)**2


def h(theta, x):
    if theta == 0:
        return x
    return theta*x
