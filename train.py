import pandas as pd
import numpy as np
import os
import shutil
import random as python_random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import time
import joblib
import tensorflow_probability as tfp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import argparse
from tqdm import tqdm
import sys
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Activation

tf.random.set_seed(2022)
np.random.seed(2022)
python_random.seed(2022)

from data.load_data import * 
from data.process_missing_data import *
from model.loss_mertric import *
from model.visualize import *
from data.data_model import *
from data.load_location import *
from model.model_2_in_1_out import *
import yaml

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Air Quality Forecasting Challenge Private')

parser.add_argument('--conf_data', default='./CONFIG/model_thien.yml', type=str, help='path of config data')
parser.add_argument('--name_model', default='thien/', type=str, help='name folder model to save')
parser.add_argument('--base_model', default='model_2', type=str, help='base model')
parser.add_argument('--batch_size', default=256, type=int, help='batch size to process data to model')
parser.add_argument('--drop_out', default=0.15, type=float, help='drop out')
parser.add_argument('--epochs', default=80, type=int, help='epochs to train')
parser.add_argument('--future', default=24, type=int, help='future step time to predict')
parser.add_argument('--n_nearest_label', default=1, type=int, help='n location nearest label data')
parser.add_argument('--n_times_choice_data', default=19, type=int, help='times choice data')
parser.add_argument('--learning_rate', default=7e-4, type=float, help='learning rate')
parser.add_argument('--n_Dense', default=64, type=int, help='n Dense in Layer Dense')
parser.add_argument('--n_LSTM', default=64, type=int, help='n LSTM in Layer LSTM')
parser.add_argument('--past', default=168, type=int, help='past step time')
parser.add_argument('--step', default=1, type=int, help='step to process data to train')
parser.add_argument('--sequence_stride', default=4, type=int, help='sequence_stride process data train X')
parser.add_argument('--sequence_stride_val', default=4, type=int, help='sequence_stride process data train Y')
parser.add_argument('--split_fraction', default=0.9, type=float, help='split data to train, val')
parser.add_argument('--stop_early', default=80, type=int, help='stop early to train')
parser.add_argument('--monitor', default='val_loss', type=str, help='monitor model val_loss or val_mdape')

args = parser.parse_args()

if __name__ == '__main__':
    # print(args.conf_path)
    with open(args.conf_data) as f:
        CONFIG_MODEL = yaml.safe_load(f)
    
    PATH_SAVE_MODEL = CONFIG_MODEL['PATH_SAVE_MODEL'] + args.name_model
    FOLDER_SUBMIT = CONFIG_MODEL['FOLDER_SUBMIT'] + args.name_model
    PATH_TRAIN_AIR = CONFIG_MODEL['PATH_TRAIN_AIR']
    PATH_TRAIN_METEO = CONFIG_MODEL['PATH_TRAIN_METEO']
    PATH_LOCATION_TRAIN_AIR = CONFIG_MODEL['PATH_LOCATION_TRAIN_AIR']
    PATH_LOCATION_TRAIN_METEO = CONFIG_MODEL['PATH_LOCATION_TRAIN_METEO']
    PATH_LOCATION_TEST_OUTPUT = CONFIG_MODEL['PATH_LOCATION_TEST_OUTPUT']

    list_meteo = CONFIG_MODEL['list_meteo']
    dict_location_input_train = CONFIG_MODEL['dict_location_input_train']

    if not os.path.exists(PATH_SAVE_MODEL):
        os.makedirs(PATH_SAVE_MODEL)
    if not os.path.exists(FOLDER_SUBMIT):
        os.makedirs(FOLDER_SUBMIT)

    print('1. LOAD DATA:')
    # Load data air
    df_total_train_input_air = load_data_air(PATH_LOCATION_TRAIN_AIR, PATH_TRAIN_AIR)
    df_total_train_input_air = df_total_train_input_air.fillna(method='pad')
    df_total_train_input_air = df_total_train_input_air.fillna(method='bfill')
    df_total_train_input_air.to_csv(PATH_SAVE_MODEL + 'train_input_air.csv')    
    # Load data meteo
    df_total_train_input_meteo = load_data_meteo(PATH_LOCATION_TRAIN_METEO, PATH_TRAIN_METEO)
    df_total_train_input_meteo.to_csv(PATH_SAVE_MODEL + 'train_input_meteo.csv')  
    # Add some data location
    df_total_train_input_air['temperature_S0000503-Times City'] = df_total_train_input_air['temperature_S0000431-Times City 2']
    df_total_train_input_air['PM2.5_S0000503-Times City'] = df_total_train_input_air['PM2.5_S0000431-Times City 2']
    df_total_train_input_air['humidity_S0000503-Times City'] = df_total_train_input_air['humidity_S0000431-Times City 2']

    df_total_train_input_air['PM2.5_S0000367-To Hieu'] = df_total_train_input_air['PM2.5_S0000509-Van Phu'] 
    df_total_train_input_air['temperature_S0000367-To Hieu'] = df_total_train_input_air['temperature_S0000509-Van Phu'] 
    df_total_train_input_air['humidity_S0000367-To Hieu'] = df_total_train_input_air['humidity_S0000509-Van Phu'] 

    df_total_train_input_air['PM2.5_S0000143-Thu vien - DHQG Ha Noi'] = df_total_train_input_air['PM2.5_S0000144-Pham Tuan Tai']
    df_total_train_input_air['temperature_S0000143-Thu vien - DHQG Ha Noi'] = df_total_train_input_air['temperature_S0000144-Pham Tuan Tai']
    df_total_train_input_air['humidity_S0000143-Thu vien - DHQG Ha Noi'] = df_total_train_input_air['humidity_S0000144-Pham Tuan Tai']
    # Merge data train air and data meteo
    list_col_train_input = get_list_col_air(list(dict_location_input_train.keys()))
    df_total_train_input_meteo_10 = df_total_train_input_meteo[list_meteo]
    df_total_train_input_air_96 = df_total_train_input_air[list_col_train_input]
    df_total_train_input = pd.merge_asof(df_total_train_input_air_96, df_total_train_input_meteo_10, left_index = True, 
                            right_index= True, direction="forward")
    df_total_train_input.to_csv(PATH_SAVE_MODEL + 'train_input.csv')
    print(f'Check null data: {df_total_train_input.isna().sum().sum()}')
    print('Load data train completed!')

    print('2. GET LABEL DATA:')
    dict_label_n_nearest = check_n_nearest_location(PATH_LOCATION_TEST_OUTPUT, PATH_LOCATION_TRAIN_AIR, args.n_nearest_label)
    df_train_output = pd.DataFrame()
    for k in dict_label_n_nearest.keys():
        col = list(dict_label_n_nearest[k].keys())
        col = ['PM2.5_' + i for i in col]
        df_train_output[k] = df_total_train_input_air[col].mean(axis= 1)
    df_train_output.to_csv(PATH_SAVE_MODEL + 'train_output.csv')

    print('3. PROCESS DATA TO TRAIN:')
    # Get X_min, X_max, Y_min, Y_max
    train_split = int(args.split_fraction * int(df_total_train_input.shape[0]))
    col_input = sorted(get_list_col_air(dict_location_input_train.keys())) + list_meteo
    dataset_train_input = df_total_train_input[col_input]
    X, X_min, X_max = normalize_min_max(dataset_train_input.values, train_split)

    dict_X_min = dict(zip(col_input, X_min.tolist()))
    dict_X_max = dict(zip(col_input, X_max.tolist()))

    Y = df_train_output.copy()
    Y, Y_min, Y_max = normalize_min_max(Y.values, train_split)

    dict_Y_min = dict(zip(df_train_output.columns.to_list(), Y_min.tolist()))
    dict_Y_max = dict(zip(df_train_output.columns.to_list(), Y_max.tolist()))

    # Choice random 8 location
    choice_8 = np.random.choice(list(dict_location_input_train.keys()), 8, 
                replace=False, p=list(dict_location_input_train.values()))
    col_input = sorted(get_list_col_air(choice_8.tolist())) + list_meteo
    dataset_train_input = df_total_train_input[col_input]

    train_split = int(args.split_fraction * int(df_total_train_input.shape[0]))
    print(f'train split: {train_split}')
    Y = df_train_output.copy()
    Y, Y_min, Y_max = normalize_min_max(Y.values, train_split)
    Y = pd.DataFrame(Y)
    print(Y.isna().sum().sum())
    X, X_min, X_max = normalize_min_max(dataset_train_input.values, train_split)
    X = pd.DataFrame(X)
    print(X.isna().sum().sum())
    train_X = X.loc[0 : train_split - 1]
    val_X = X.loc[train_split:]

    start = args.past
    end = args.future + train_split
    sequence_length_x = int(args.past / args.step)
    sequence_length_y = int(args.future/ args.step)
    print(f'train start: {start}, end: {end}')
    train_Y = Y.iloc[start:end]

    train_dataset = create_tf_data_2_in_1_out(train_X , 
                train_Y ,
                sequence_stride = args.sequence_stride,
                sequence_length_x = sequence_length_x,
                sequence_length_y = sequence_length_y,
                sampling_rate= args.step,
                batch_size = args.batch_size,
                shuffle= True)
    for batch in train_dataset.take(1):
        train_input, train_output = batch
        train_input_1, train_input_2 = train_input
        print(f'train inputs 1 shape: {train_input_1.shape}; train inputs 2 shape: {train_input_2.shape}; train outputs shape: {train_output.shape}')
        break

    x_end = len(val_X) - args.future
    label_start = train_split + args.past

    print(f'val label start: {label_start}, x end: {x_end}')
    val_Y = Y.iloc[label_start:]
    val_X_ = val_X.iloc[:x_end]
    print(f'val X shape: {val_X.shape}, val Y shape: {val_Y.shape}')

    val_dataset = create_tf_data_2_in_1_out(val_X_ , 
                val_Y ,
                sequence_stride = args.sequence_stride,
                sequence_length_x = sequence_length_x,
                sequence_length_y = sequence_length_y,
                sampling_rate= args.step,
                batch_size = args.batch_size,
                shuffle=False)

    for batch in val_dataset.take(1):
        val_input, val_output = batch
        val_input_1, val_input_2 = val_input
        print(f'val inputs shape: {val_input_1.shape, val_input_2.shape}; val outputs shape: {val_output.shape}')
        break

    # Repeat choice data
    for i in range(args.n_times_choice_data):
        choice_8 = np.random.choice(list(dict_location_input_train.keys()), 8, 
                        replace=False, p=list(dict_location_input_train.values()))
        col_input = sorted(get_list_col_air(choice_8.tolist())) + list_meteo
        # print(col_input)
        dataset_train_input = df_total_train_input[col_input]

        train_split = int(args.split_fraction * int(df_total_train_input.shape[0]))
        Y = df_train_output.copy()
        Y, Y_min, Y_max = normalize_min_max(Y.values, train_split)
        Y = pd.DataFrame(Y)
        X, X_min, X_max = normalize_min_max(dataset_train_input.values, train_split)
        X = pd.DataFrame(X)
        train_X = X.loc[0 : train_split - 1]
        val_X = X.loc[train_split:]

        start = args.past
        end = args.future + train_split
        sequence_length_x = int(args.past / args.step)
        sequence_length_y = int(args.future/ args.step)
        train_Y = Y.iloc[start:end]

        train_dataset_i = create_tf_data_2_in_1_out(train_X , 
                    train_Y ,
                    sequence_stride = args.sequence_stride,
                    sequence_length_x = sequence_length_x,
                    sequence_length_y = sequence_length_y,
                    sampling_rate= args.step,
                    batch_size = args.batch_size,
                    shuffle= True)
        train_dataset = train_dataset.concatenate(train_dataset_i)
        
        x_end = len(val_X) - args.future
        label_start = train_split + args.past

        val_Y = Y.iloc[label_start:]
        val_X_ = val_X.iloc[:x_end]

        val_dataset_i = create_tf_data_2_in_1_out(val_X_ , 
                    val_Y ,
                    sequence_stride = args.sequence_stride,
                    sequence_length_x = sequence_length_x,
                    sequence_length_y = sequence_length_y,
                    sampling_rate= args.step,
                    batch_size = args.batch_size,
                    shuffle=False)
        val_dataset = val_dataset.concatenate(val_dataset_i)

    print('4. TRAIN DATA:')
    input1_shape_0, input1_shape_1 = train_input_1.shape[1:]
    input2_shape_0, input2_shape_1 = train_input_2.shape[1:]
    output_shape_0, output_shape_1 = train_output.shape[1:]

    if args.base_model == 'model_1':
        model = model_2_in_1_out(
            input1_shape_0 = input1_shape_0,
            input1_shape_1 = input1_shape_1,
            input2_shape_0 = input2_shape_0,
            input2_shape_1 = input2_shape_1,
            output_shape_0 = output_shape_0,
            output_shape_1 = output_shape_1,
            drop_out = args.drop_out,
            n_LSTM = args.n_LSTM,
            n_Dense = args.n_Dense,
            )
    elif args.base_model == 'model_2':
        model = model_2_in_1_out_2(
            input1_shape_0 = input1_shape_0,
            input1_shape_1 = input1_shape_1,
            input2_shape_0 = input2_shape_0,
            input2_shape_1 = input2_shape_1,
            output_shape_0 = output_shape_0,
            output_shape_1 = output_shape_1,
            drop_out = args.drop_out,
            n_LSTM = args.n_LSTM,
            n_Dense = args.n_Dense,
            )
    print(model.summary())

    metrics = [
        mdape,
        tf.keras.metrics.MeanAbsoluteError(name = 'MAE'),
        tf.keras.metrics.RootMeanSquaredError(name = 'RMSE')
        ]
    optimizer = tf.keras.optimizers.RMSprop(learning_rate= args.learning_rate, clipnorm=10)
    model.compile(loss= 'mae',
                optimizer=optimizer,
                metrics= metrics)
    model_checkpoint = keras.callbacks.ModelCheckpoint(PATH_SAVE_MODEL+"my_checkpoint.h5", 
            save_best_only=True,monitor= args.monitor , verbose=1)
    early_stopping = keras.callbacks.EarlyStopping(patience= args.stop_early, monitor= args.monitor)
    print('START TRAIN MODEL:')
    history = model.fit(train_dataset, epochs= args.epochs,
            validation_data= val_dataset,
            callbacks=[early_stopping, model_checkpoint], verbose= 1, batch_size= args.batch_size)
    
    print('5.EVALUATE MODEL:')
    visualize_loss(history, "Training and Validation Loss", PATH_SAVE_MODEL)
    visualize_mdape(history, "Training and Validation mdape", PATH_SAVE_MODEL)

    model = keras.models.load_model(PATH_SAVE_MODEL+"my_checkpoint.h5", custom_objects={"mdape": mdape, 'loss_custom':loss_custom })
    print(model.evaluate(train_dataset))
    print(model.evaluate(val_dataset))

    dict_model = dict( 
        name = args.name_model,
        base_model = args.base_model,
        Args = dict (
            batch_size= args.batch_size,
            drop_out= args.drop_out,
            epochs= args.epochs,
            future= args.future,
            learning_rate= args.learning_rate,
            n_Dense= args.n_Dense,
            n_LSTM= args.n_LSTM,
            past= args.past,
            sequence_stride= args.sequence_stride,
            sequence_stride_val= args.sequence_stride_val,
            split_fraction= args.split_fraction,
            step= args.step,
            stop_early= args.stop_early,
            monitor= args.monitor,
            n_times_choice_data = args.n_times_choice_data,
            n_nearest_label = args.n_nearest_label
            ),
        X_min = dict_X_min,
        X_max = dict_X_max,
        Y_min = dict_Y_min,
        Y_max = dict_Y_max,
        list_meteo = list_meteo,
        PATH_LOCATION_TEST_OUTPUT = CONFIG_MODEL['PATH_LOCATION_TEST_OUTPUT'],
        path_model = PATH_SAVE_MODEL+"my_checkpoint.h5",
        path_save_model_yaml = PATH_SAVE_MODEL +'model_save.yml'
    )

    with open(PATH_SAVE_MODEL +'model_save.yml', 'w') as yaml_file:
        yaml.dump(dict_model, yaml_file, default_flow_style=False)