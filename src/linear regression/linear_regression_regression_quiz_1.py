import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def random_numbers():
    X = np.arange(-3, 3, 0.05)
    Y = [y(x) for x in X]
    return (X, Y)


def y(x):
    return x**2 + 0.5*x + 1 + np.random.uniform(-0.5, .5, 1)


def fit_with_model(regressor, X, Y):
    regressor.fit(np.c_[X], np.c_[Y])
    predicted_y = regressor.predict(np.c_[X])
    return predicted_y


if __name__ == "__main__":
    (X, Y) = random_numbers()

    linear_reg = LinearRegression()
    linear_reg_y = fit_with_model(linear_reg, X, Y)
    print(linear_reg.coef_)
    print("MSE: {}".format(mean_squared_error(Y, linear_reg_y)))
    plt.scatter(X, Y)
    plt.scatter(X, linear_reg_y)
    plt.show()

    sgd_reg = SGDRegressor()
    sgd_reg_y = fit_with_model(sgd_reg, X, Y)
    print(sgd_reg.coef_)
    print("MSE: {}".format(mean_squared_error(Y, sgd_reg_y)))
    plt.scatter(X, Y)
    plt.scatter(X, sgd_reg_y)
    plt.show()

    dt_reg = DecisionTreeRegressor()
    dt_reg_y = fit_with_model(dt_reg, X, Y)
    print("MSE: {}".format(mean_squared_error(Y, dt_reg_y)))
    plt.scatter(X, Y)
    plt.scatter(X, dt_reg_y)
    plt.show()
