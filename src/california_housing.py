import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = pd.read_csv('./src/calHouse.csv')

def investigate_ds():
    print(data.info())
    op_distinct = data['ocean_proximity'].value_counts()
    print("distinct op: {}".format(op_distinct))
    print(data.columns)
def transform_enums_to_number():
    # transform enums to numbers (manual)
    data2 = data
    data2.replace({'<1H OCEAN': 1, 'INLAND': 2, 'NEAR OCEAN': 3,
                   'NEAR BAY': 4, 'ISLAND': 5}, inplace=True)
    print(data2)

    # transform enums to number (sscikit learn)
    data3 = data
    encoder = LabelEncoder()
    encoder.fit(data3['ocean_proximity'])
    data3['ocean_proximity'] = encoder.transform(data3['ocean_proximity'])
    print(data3)
def plot():
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

    # conflicts with above  plt
    correlation = data.corr()
    sb.heatmap(correlation)
    plt.show()

lats = data['longitude'].values
longs = data['latitude'].values
investigate_ds()
transform_enums_to_number()
plot()
