import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./src/country_stat.csv')
gdp_per_capita = data['GDP per capita']
life_satisfaction = data['Life satisfaction']
plt.scatter(gdp_per_capita, life_satisfaction)
plt.show()
