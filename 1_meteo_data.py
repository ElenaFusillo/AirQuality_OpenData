import os
import pandas as pd
from sklearn.impute import KNNImputer


#METEO DATA, downloaded from the Dexter website and manipulated before using this code

root = os.getcwd()
my_path_meteo = root+"/Meteo_data"
my_path_RN_data = root+"/RN_data"

files_meteo = os.scandir(my_path_meteo)

if not os.path.exists(my_path_RN_data):
    os.makedirs(my_path_RN_data)

df_meteo = pd.DataFrame()

for file_meteo in files_meteo:
    df = pd.read_csv(file_meteo, parse_dates=[0], dayfirst=True,usecols=[0,2])
    df.set_index('DATA_INIZIO', inplace=True)
    df_meteo = pd.concat([df_meteo, df], axis=1)
df_meteo.to_csv(my_path_meteo+'/RN_meteo_hour.csv')

RN_meteo_hour = pd.read_csv(my_path_meteo+'/RN_meteo_hour.csv', parse_dates=[0], dayfirst=True)
RN_meteo_hour.set_index('DATA_INIZIO', inplace=True)

#There are some null values

imputing_cols = RN_meteo_hour.columns
knn_imputer = KNNImputer(n_neighbors = 5,
                         weights='uniform',
                         metric='nan_euclidean')
imputed = knn_imputer.fit_transform(RN_meteo_hour[imputing_cols])
RN_meteo_hour.loc[:, imputing_cols] = imputed.round(2) # to replace the original columns with the ones imputed
RN_meteo_hour.to_csv(my_path_meteo+'/RN_meteo_hour_5kNN_filled.csv')