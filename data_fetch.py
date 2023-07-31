import pandas as pd
import numpy
import os


def match_digits(lut, digits, column, lista=[]):
    '''Simple function that matches the lenght of the .csv file name'''
    lista = sorted(lut[column].drop_duplicates())
    for i in range(len(lista)):
        lista[i] = str(lista[i]).rjust(digits, '0')
    return lista

def create_ref_df(year, freq=['D','H']):
    '''Function that creates a reference dataframe in order to join all the .csv files'''
    if freq=='D':
        mult = 1
        num_periods = (366 if (year in [2012, 2016, 2020]) else 365)*mult
        ref_time_inizio = pd.Series(pd.date_range(start=str(year)+'/01/01', periods=num_periods, freq=freq))
        ref_dict = {'DATA_INIZIO':ref_time_inizio}
        ref_df = pd.DataFrame(data=ref_dict)
    elif freq=='H':
        mult = 24
        num_periods = (366 if (year in [2012, 2016, 2020]) else 365)*mult
        ref_time_inizio = pd.Series(pd.date_range(start=str(year)+'/01/01 00:00:00', periods=num_periods, freq=freq))
        ref_dict = {'DATA_INIZIO':ref_time_inizio}
        ref_df = pd.DataFrame(data=ref_dict)
    else:
        raise Exception('Wrong frequency, only D or H frequency are valid.')
    return ref_df

def get_stations_and_parameters(df):
    stations = set()
    parameters = set()
    for colname in list(df.columns):
        station, parameter = colname.split("_")
        stations.add(int(station))
        parameters.add(int(parameter))
    return stations, parameters

def load_homemade_csv(folderName="RN_data"):
    df = None
    for file in os.listdir(folderName):
        if file.endswith(".csv"):
            partial_df = pd.read_csv(os.path.join(folderName, file), parse_dates=[0], dayfirst=True)
            partial_df['DATA_INIZIO'] = pd.to_datetime(partial_df['DATA_INIZIO'])
            partial_df.set_index(['DATA_INIZIO'], inplace=True)
            df = partial_df.copy()

    stations, parameters = get_stations_and_parameters(df)
    
    header = pd.MultiIndex.from_product([stations,
                                        parameters],
                                        names=['Station','Parameter'])
    df_res = pd.DataFrame(columns=header, dtype=float)
    df_res['DATA_INIZIO'] = df.index.copy()
    df_res.set_index(['DATA_INIZIO'], inplace=True)

    for colname in list(df.columns):
        station, parameter = colname.split("_")
        df_res[int(station), int(parameter)] = df[colname]

    return df_res