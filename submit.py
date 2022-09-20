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


parser = argparse.ArgumentParser(description='Air Quality Forecasting Challenge Private')
parser.add_argument('--path_save_submit', default='./submit/', type=str, help='path save file result csv')
parser.add_argument('--path_data_test', default='./data_test/input/', type=str, help='path folder publich test')
parser.add_argument('--conf_model', default='./save_model/model_6/model_save.yml', type=str, help='name folder model to save')
args = parser.parse_args()

if __name__ == '__main__':
 
    with open(args.conf_model) as f:
        CONFIG_MODEL = yaml.safe_load(f)
    
    name_model = CONFIG_MODEL['name']
    FOLDER_SUBMIT = args.path_save_submit + name_model
    TEST_INPUT_PATH = args.path_data_test
    dict_X_min = CONFIG_MODEL['X_min']
    dict_X_max = CONFIG_MODEL['X_max']
    dict_Y_min = CONFIG_MODEL['Y_min']
    dict_Y_max = CONFIG_MODEL['Y_max']
    dict_Y_min = np.array(list(dict_Y_min.values()))
    dict_Y_max = np.array(list(dict_Y_max.values()))
    list_meteo = CONFIG_MODEL['list_meteo']
    PATH_LOCATION_TEST_OUTPUT = CONFIG_MODEL['PATH_LOCATION_TEST_OUTPUT']
    PATH_MODEL = CONFIG_MODEL['path_model']
    model = keras.models.load_model(PATH_MODEL, custom_objects={"mdape": mdape,'loss_custom':loss_custom })

    for i in range(1,90):
        file_location_air = os.path.join(TEST_INPUT_PATH, str(i), 'location_input.csv')
        path_folder_air = os.path.join(TEST_INPUT_PATH, str(i))
        np_air_i = load_data_air_test(file_location_air, path_folder_air, dict_X_min, dict_X_max)
        
        file_location_meteo = os.path.join(TEST_INPUT_PATH, str(i),'meteo', 'location_meteorology.csv')
        path_folder_meteo = os.path.join(TEST_INPUT_PATH, str(i),'meteo')
        np_meteo_i = load_data_meteo_test(file_location_meteo, path_folder_meteo,list_meteo, dict_X_min, dict_X_max)

        np_air_i = np.expand_dims(np_air_i, axis=0)
        np_meteo_i = np.expand_dims(np_meteo_i, axis=0)

        predict = model.predict([np_air_i, np_meteo_i ])
        predict = np.squeeze(predict, axis= 0)
        predict = (predict * (dict_Y_max - dict_Y_min)) + dict_Y_min

        file_location_out = os.path.join(TEST_INPUT_PATH, str(i), 'location_output.csv')

        dict_order = get_order_submit(file_location_out, PATH_LOCATION_TEST_OUTPUT)

        if not os.path.exists(FOLDER_SUBMIT + str(i)):
            os.makedirs(FOLDER_SUBMIT + str(i))

        for k,v in dict_order.items():
            name_submit = FOLDER_SUBMIT + str(i) + '/res_'+ str(i)+ '_' +str(k) + '.csv'
            pd.DataFrame(predict[:,v], columns= ['PM2.5']).to_csv(name_submit, index= False)

    shutil.make_archive('submit' + name_model[:-1] , 'zip', FOLDER_SUBMIT)