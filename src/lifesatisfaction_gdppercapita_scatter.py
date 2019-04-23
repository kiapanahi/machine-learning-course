import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data():
    data = pd.read_csv('./src/country_stat.csv')
    return data


def plot_life_sat_on_gdp():
    data = load_data()
    gdp_per_capita = data['GDP per capita']
    life_satisfaction = data['Life satisfaction']
    plt.scatter(gdp_per_capita, life_satisfaction)
    plt.show()


def create_linear_regression_model():
    lr_model = LinearRegression()
    data = load_data()
    gdp_per_capita = data['GDP per capita']
    life_satisfaction = data['Life satisfaction']
    X = np.c_[gdp_per_capita.values]
    y = np.c_[life_satisfaction.values]

    lr_model.fit(X, y)

    y_predictions = lr_model.predict(X)

    plt.scatter(X, y)
    plt.plot(X, y_predictions)
    plt.show()


if __name__ == "__main__":
    # plot_life_sat_on_gdp()
    create_linear_regression_model()
