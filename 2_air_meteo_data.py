import os
import numpy as np
import pandas as pd


root = os.getcwd()
my_path = root+"/Air_data/"
my_path_day = my_path+"/DAY/"
my_path_hour = my_path+"/HOUR/"
my_path_day_knn = my_path +"/DAY_5kNN_filled/"
my_path_hour_knn = my_path +"/HOUR_5kNN_filled/"
my_path_meteo = root + "/Meteo_data"
my_path_RN_data = root+"/RN_data"

if not os.path.exists(my_path_RN_data):
    os.makedirs(my_path_RN_data)

ALL_air_hour = pd.read_csv(my_path+"ALL_stations_hour.csv", parse_dates=[0], dayfirst=True)
ALL_air_hour.set_index('DATA_INIZIO', inplace=True)

RN_meteo_hour = pd.read_csv(my_path_meteo+'/RN_meteo_hour_5kNN_filled.csv', parse_dates=[0], dayfirst=True)
RN_meteo_hour.set_index('DATA_INIZIO', inplace=True)

COD_STAZ_RN = ['10000001', '10000060', '10000074']
PARAM_RN = ['8', '1001', '1002', '1003', '1004', '1005'] #NO2 and METEO parameters

RN_air_hour = pd.DataFrame()

for colname in list(ALL_air_hour.columns):
    station, parameter = colname.split("_")
    if (station in COD_STAZ_RN) & (parameter in PARAM_RN):
        RN_air_hour[colname] = ALL_air_hour[colname]

RN_meteo_hour.index = RN_meteo_hour.index.tz_localize(tz=None)

RN_air_meteo_data = pd.concat([RN_air_hour, RN_meteo_hour], axis=1)
RN_air_meteo_data = RN_air_meteo_data.round(2)

RN_air_meteo_data.to_csv(my_path_RN_data+"\RN_air_meteo_data.csv")
