import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def load_data_air(path_location, path_folder):
    """
    Load data train input air (In example: data of 71 station)
    Args:
        path_location: dir path location file --> read satation order
        path_folder: dir path of folder contain data train input or data train output
    Return:
        file csv data combine train input or output
    """
    df_total_train = pd.DataFrame()
    for i in tqdm(pd.read_csv(path_location)['location'].map(lambda x: str(x)+ '.csv').to_list()):
        try:
            df = pd.read_csv(os.path.join(path_folder, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_total_train.columns) < 2:
                df_total_train['time'] = df['time']
            df_total_train = df_total_train.merge(df, on = 'time')
        except:
            print(f'No found file: {i} !')
    # df_total_train['time'] = pd.to_datetime(df_total_train['time'], dayfirst= True)
    df_total_train['time'] = pd.to_datetime(df_total_train['time'])
    df_total_train = df_total_train.set_index('time')

    return df_total_train

def load_data_meteo(path_location, path_folder):
    """
    Load data train input meteo (In example: data of 143 station)
    Args:
        path_location: dir path location file --> read satation order
        path_folder: dir path of folder contain data train input or data train output
    Return:
        file csv data combine train input or output
    """
    df_total_train = pd.DataFrame()
    for i in tqdm(pd.read_csv(path_location)['stat_name'].map(lambda x: str(x)+ '.csv').to_list()):
        try:
            df = pd.read_csv(os.path.join(path_folder, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            df['ws'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)
            df['direction'] = np.arctan2(df['v10'], df['u10'])
            df = df[['time','ws','direction', 'total_precipitation','surface_pressure','evaporation' ]]

            dict_col_change = {
            # 'u10': 'u10_' + name_location_i,
            # 'v10': 'v10_'+ name_location_i,
            'total_precipitation': 'total_precipitation_'+ name_location_i,
            'surface_pressure': 'surface_pressure_'+ name_location_i,
            'evaporation': 'evaporation_'+ name_location_i,
            'ws': 'ws_'+ name_location_i,
            'direction': 'direction_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_total_train.columns) < 2:
                df_total_train['time'] = df['time']
            df_total_train = df_total_train.merge(df, on = 'time')
        except:
            print(f'No found file: {i} !')
    df_total_train['time'] = pd.to_datetime(df_total_train['time'])
    # df_total_train['time'] = pd.to_datetime(df_total_train['time'], dayfirst= True)
    df_total_train = df_total_train.set_index('time')

    return df_total_train

def check_feature(df, p):
    """
    Keep column of data df has missing data percent < p%
    Args:
        df: DataFrame
        p: percent
    Return:
        List columns have data minsing/ total < p%
    """
    k = 0
    list_feature = []
    for i in df.columns:
        n_miss = df[[i]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        if perc.values[0]< p:
            # print('> %s, Missing: %d (%.1f%%)' % (i, n_miss, perc))
            k += 1
            list_feature.append(i)
    print(f'feature has percent missing data < {p}%: {k} / total: {len(df.columns)} ~ {(k/len(df.columns) *100):.2f} %')
    return list_feature

def normalize_min_max(data, train_split):
    # Normalize Min-Max data
    data_min = data[:train_split].min(axis=0)
    data_max = data[:train_split].max(axis=0)
    return (data - data_min) / (data_max - data_min) , data_min, data_max

def normalize_mean_std(data, train_split):
    # Normalize Mean-std data
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std , data_mean, data_std

def make_date(dataframe, list_add_date = ['year', 'month', 'day', 'hour', 'day_of_week']):
    COL_NAME = dataframe.columns.to_list()
    df = dataframe.reset_index('time')
    df['year'] = df.time.dt.year
    df['month'] = df.time.dt.month
    df['day'] = df.time.dt.month
    df['hour'] = df.time.dt.hour
    df['day_of_week'] = df.time.dt.dayofweek
    df = df.set_index('time')
    df = df[COL_NAME + list_add_date]
    return df

def get_list_col_air(list_col):
    list_col_train_input = []
    for i in list_col:
        list_col_train_input.append('PM2.5_' + i)
        list_col_train_input.append('humidity_' + i)
        list_col_train_input.append('temperature_' + i)
    return list_col_train_input

def load_data_test(path_folder, col_name):
    """
    Load data test to predict and submit
    Args:
        path_folder: dir path of data test
        col_name(list): list columns order by data train
    Return:
        dict data of each folder in path_folder test.
    """
    dict_data_submit = {}
    for k in tqdm(os.listdir(path_folder)):
        PUBLIC_TEST_INPUT_PATH_FOLDER = os.path.join(path_folder, k)
        df_concat = pd.DataFrame()
        for i in os.listdir(PUBLIC_TEST_INPUT_PATH_FOLDER):
            df = pd.read_csv(os.path.join(PUBLIC_TEST_INPUT_PATH_FOLDER, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_concat.columns) < 2:
                df_concat['timestamp'] = df['timestamp']
            df_concat = df_concat.merge(df, on = 'timestamp')
        value_k = df_concat[col_name].values
        dict_data_submit[k] = value_k
    return dict_data_submit

def load_data_air_test(path_location, path_folder, dict_min, dict_max):
    """
    Load data train input air (In example: data of 71 station)
    Args:
        path_location: dir path location file --> read satation order
        path_folder: dir path of folder contain data train input or data train output
    Return:
        file csv data combine train input or output
    """
    df_total_train = pd.DataFrame()
    for i in tqdm(pd.read_csv(path_location)['location'].map(lambda x: str(x)+ '.csv').to_list()):
        try:
            df = pd.read_csv(os.path.join(path_folder, i))
            df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            # name_location_i = dict_change_name[name_location_i]
            dict_col_change = {
            'PM2.5': 'PM2.5_' + name_location_i,
            'humidity': 'humidity_'+ name_location_i,
            'temperature': 'temperature_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_total_train.columns) < 2:
                df_total_train['time'] = df['time']
            df_total_train = df_total_train.merge(df, on = 'time')
        except:
            print(f'No found file: {i} !')
    # df_total_train['time'] = pd.to_datetime(df_total_train['time'], dayfirst= True)
    df_total_train['time'] = pd.to_datetime(df_total_train['time'])
    df_total_train = df_total_train.set_index('time')
    col_name = sorted(df_total_train.columns.to_list())
    list_min = np.array([dict_min[i] for i in col_name])
    list_max = np.array([dict_max[i] for i in col_name])
    df_total_train = df_total_train[col_name]
    df_total_train = df_total_train.fillna(method='pad')
    df_total_train = df_total_train.fillna(method='bfill')
    if df_total_train.isna().sum().sum() > 0:
        for i in col_name:
            if df_total_train[i].isna().sum().sum() > 0:
                df_total_train[i] = (dict_max[i] + dict_min[i])/2.
    np_air = (df_total_train.to_numpy() - list_min)/ (list_max - list_min)

    return np_air

def load_data_meteo_test(path_location, path_folder,col_name, dict_min, dict_max):
    """
    Load data train input meteo (In example: data of 143 station)
    Args:
        path_location: dir path location file --> read satation order
        path_folder: dir path of folder contain data train input or data train output
    Return:
        file csv data combine train input or output
    """
    df_total_train = pd.DataFrame()
    for i in tqdm(pd.read_csv(path_location)['stat_name'].map(lambda x: str(x)+ '.csv').to_list()):
        try:
            df = pd.read_csv(os.path.join(path_folder, i))
            # df.drop(['Unnamed: 0'], axis= 1, inplace = True)
            name_location_i = i.split('.')[0]
            df['ws'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)
            df['direction'] = np.arctan2(df['v10'], df['u10'])
            df = df[['time','ws','direction', 'total_precipitation','surface_pressure','evaporation' ]]

            dict_col_change = {
            # 'u10': 'u10_' + name_location_i,
            # 'v10': 'v10_'+ name_location_i,
            'total_precipitation': 'total_precipitation_'+ name_location_i,
            'surface_pressure': 'surface_pressure_'+ name_location_i,
            'evaporation': 'evaporation_'+ name_location_i,
            'ws': 'ws_'+ name_location_i,
            'direction': 'direction_'+ name_location_i
                }
            df.rename(columns= dict_col_change, inplace= True)
            if len(df_total_train.columns) < 2:
                df_total_train['time'] = df['time']
            df_total_train = df_total_train.merge(df, on = 'time')
        except:
            print(f'No found file: {i} !')
    df_total_train['time'] = pd.to_datetime(df_total_train['time'])
    # df_total_train['time'] = pd.to_datetime(df_total_train['time'], dayfirst= True)
    df_total_train = df_total_train.set_index('time')
    df_total_train = df_total_train[col_name]
    df_total_train = df_total_train[:56]
    list_min = np.array([dict_min[i] for i in col_name])
    list_max = np.array([dict_max[i] for i in col_name])

    np_meteo = (df_total_train.to_numpy() - list_min)/ (list_max - list_min)

    return np_meteo

def get_order_submit(path_out, path_csv10):
    df1 = pd.read_csv(path_csv10)
    df1.longitude = df1.longitude.map(lambda x: round(x,4))
    df1.latitude = df1.latitude.map(lambda x: round(x,4))
    df1['key'] = df1.longitude.astype(str) + df1.latitude.astype(str)
    
    df = pd.read_csv(path_out)
    df.longitude = df.longitude.map(lambda x: round(x,4))
    df.latitude = df.latitude.map(lambda x: round(x,4))
    df['key'] = df.longitude.astype(str) + df.latitude.astype(str)

    df = df.merge(df1, on = 'key')
    df.location_y = df.location_y.map(lambda x: int(x[-1]))
    dict_order = dict(zip(df.location_x.to_list(), df.location_y.to_list()))

    return dict_order