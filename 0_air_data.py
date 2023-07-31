import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from data_fetch import match_digits, create_ref_df


root = os.getcwd()
my_path = root+"/Air_data/"
my_path_day = my_path+"/DAY/"
my_path_hour = my_path+"/HOUR/"
my_path_day_knn = my_path +"/DAY_5kNN_filled/"
my_path_hour_knn = my_path +"/HOUR_5kNN_filled/"

if not os.path.exists(my_path_day):
    os.makedirs(my_path_day)

if not os.path.exists(my_path_hour):
    os.makedirs(my_path_hour)

if not os.path.exists(my_path_day_knn):
    os.makedirs(my_path_day_knn)

if not os.path.exists(my_path_hour_knn):
    os.makedirs(my_path_hour_knn)


INDEX_INQ_DAY = [1, 11] #daily pollutants indexes actually inside .csv files
INDEX_INQ_HOUR = [0, 2, 3, 4, 5, 6, 7, 9, 10] #hourly pollutants indexes actually inside .csv files


#LOAD THE LOOK UP TABLES
params_lut = pd.read_excel(root+'/parametri.xlsx', usecols="A:D", dtype={'IdParametro': np.int16, 'Tmed (min)': np.int16})
stations_lut = pd.read_excel(root+'/QARIA_Stazioni.xlsx', dtype={'Cod_staz':str, 'Id_Param': str})

Cod_staz = match_digits(stations_lut, 8, column='Cod_staz') #list of strings, 54 elements
IdParametro = match_digits(params_lut, 3, column='IdParametro') #list of strings, 21 elements

#Remove the local stations and the fictitous ones
staz_to_remove = ['02000229','02000230', '02000232', '05000020', '05000024', '07000046', '07000047']
for i in staz_to_remove:
    Cod_staz.remove(i)


#LOAD THE DATA: daily values loop, quick (25s)
for station in range(0, len(Cod_staz)):
    choosen_station = Cod_staz[station]
    result_day = pd.DataFrame()
    for year in range(2010, 2023):
        ref_df_day = create_ref_df(year, freq='D')
        for index in INDEX_INQ_DAY:
            name_file_import = my_path + "storico_"+str(year)+'_'+choosen_station+'_'+IdParametro[index]+'.csv'
            if not os.path.exists(name_file_import):
                continue
            my_data = pd.read_csv(name_file_import, parse_dates=[1, 2], dayfirst=True, usecols=[0,1,2,4])
            name_of_column = str(my_data.COD_STAZ.unique()[0]) + '_' + str(my_data.ID_PARAM.unique()[0])
            my_data.rename(columns={'VALORE': name_of_column}, inplace=True)
            my_data.drop(['COD_STAZ', 'ID_PARAM'], axis=1, inplace=True)
            ref_df_day = pd.merge(ref_df_day, my_data, on=['DATA_INIZIO'], how='left')
        result_day = pd.concat([result_day, ref_df_day])
    result_day.to_csv(my_path_day+choosen_station+'_day.csv', index=False)


#LOAD THE DATA: hourly values loop, slow (20min)

hour_Cod_staz = Cod_staz.copy()
hour_Cod_staz.remove('10000002') #no hourly data present!

for station in range(0, len(hour_Cod_staz)):
    choosen_station = hour_Cod_staz[station]
    result_hour = pd.DataFrame()
    for year in range(2010, 2023):
        ref_df_hour = create_ref_df(year, freq='H')
        for index in INDEX_INQ_HOUR:
            name_file_import = my_path +"storico_"+str(year)+'_'+choosen_station+'_'+IdParametro[index]+'.csv'
            if not os.path.exists(name_file_import):
                continue
            my_data = pd.read_csv(name_file_import, parse_dates=[1, 2], dayfirst=True, usecols=[0,1,2,4])
            name_of_column = str(my_data.COD_STAZ.unique()[0]) + '_' + str(my_data.ID_PARAM.unique()[0])
            my_data.rename(columns={'VALORE':name_of_column}, inplace=True)
            my_data.drop(['COD_STAZ', 'ID_PARAM'], axis=1, inplace=True)
            ref_df_hour = pd.merge(ref_df_hour, my_data, on=['DATA_INIZIO'], how='left')
        result_hour = pd.concat([result_hour, ref_df_hour])
    result_hour.to_csv(my_path_hour+choosen_station+'_hour.csv', index=False)


folder_day = os.scandir(my_path_day)
folder_hour = os.scandir(my_path_hour)

for file_day in folder_day:
    df = pd.read_csv(file_day, parse_dates=True, dayfirst=True)
    imputing_cols = df.columns.drop(['DATA_INIZIO'])
    knn_imputer = KNNImputer(n_neighbors = 5,
                             weights='uniform',
                             metric='nan_euclidean')
    imputed = knn_imputer.fit_transform(df[imputing_cols])
    df.loc[:, imputing_cols] = imputed.round(1)
    file_name = os.path.splitext(file_day)[0].split("/")[-1]
    new_path = my_path_day_knn+file_name+"_5kNN_filled.csv"
    df.to_csv(new_path, index=False)

for file_hour in folder_hour:
    df = pd.read_csv(file_hour, parse_dates=True, dayfirst=True)
    imputing_cols = df.columns.drop(['DATA_INIZIO'])# it works for any number of columns
    knn_imputer = KNNImputer(n_neighbors = 5,
                             weights='uniform',
                             metric='nan_euclidean')
    imputed = knn_imputer.fit_transform(df[imputing_cols])
    df.loc[:, imputing_cols] = imputed.round(1) # to replace the original columns with the ones imputed
    file_name = os.path.splitext(file_hour)[0].split("/")[-1]
    new_path = my_path_hour_knn+file_name+"_5kNN_filled.csv"
    df.to_csv(new_path, index=False)


folder_day_knn = os.scandir(my_path_day_knn)
folder_hour_knn = os.scandir(my_path_hour_knn)


result_day = pd.DataFrame()

for file_day_knn in folder_day_knn:
    doc = pd.read_csv(file_day_knn, parse_dates=[0], dayfirst=True)
    doc.set_index('DATA_INIZIO', inplace=True)
    result_day = pd.concat([result_day, doc], axis=1)
result_day.to_csv(my_path+'/ALL_stations_day.csv')

result_hour = pd.DataFrame()

for file_hour_knn in folder_hour_knn:
    doc = pd.read_csv(file_hour_knn, parse_dates=[0], dayfirst=True)
    doc.set_index('DATA_INIZIO', inplace=True)
    result_hour = result_hour.loc[~result_hour.index.duplicated(keep='first')]
    result_hour = pd.concat([result_hour, doc], axis=1)
result_hour.to_csv(my_path+'/ALL_stations_hour.csv')