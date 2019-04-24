import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

data = pd.read_csv('./src/calHouse.csv')

# investigate data structure and attributes
print(data.info())
op_distinct = data['ocean_proximity'].value_counts()
print("distinct op: {}".format(op_distinct))

print(data.columns)
lats = data['longitude'].values
longs = data['latitude'].values


plt.scatter(lats, longs,
            alpha=0.4,
            c=data['median_house_value'],
            s=data['population']*0.01,
            label='Population')
plt.colorbar()
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

correlation = data.corr()
sb.heatmap(correlation)
