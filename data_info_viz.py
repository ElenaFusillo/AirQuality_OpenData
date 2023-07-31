import pandas as pd
from data_fetch import load_homemade_csv
import matplotlib.pyplot as plt

df = load_homemade_csv(folderName="RN_data")

print(df.head())
print(df.tail())
print(df.describe())


#df.loc[:, df.columns.get_level_values(0).isin({10000001})]
#df.loc[:, df.columns.get_level_values(0).isin({10000059})]
#df.loc[:, df.columns.get_level_values(0).isin({10000074})]


def print_graphs_RNN():
    parameters = list(df.columns.levels[1][:-1])
    stations = list(df.columns.levels[0][:-1])
    for parameter in parameters:
        for station in stations:
            ax = df[station][parameter].plot(figsize=(12, 8), title="Station: {0} - Parameter: {1}".format(station, parameter))
            #ax.set_ylabel("NO2 [ug/m3]")
            plt.show()

print_graphs_RNN()

