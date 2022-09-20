import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import os
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline

def KNN_Imputer(data, n_step, n_neighbors, directory_save_model):
    """
    Process missing data by KNN Imputer
    Args:
        data: Data missing
        n_step: slice data to n_step part
        n_neighbors: n_neighbers in KNN
        directory_save_model: dir save model KNN_Imputer
    Return:
        list file dir path of model Imputer
    """
    len_data = data.shape[0]
    slice_data = np.arange(len_data)
    slice_data = np.array(np.split(slice_data, len_data//n_step))
    slice_data = np.reshape(slice_data, len_data, order='F')
    slice_data = np.split(slice_data, n_step)
    list_model_imputer = []
    if not os.path.exists(directory_save_model):
        os.makedirs(directory_save_model)
    for i in range(len(slice_data)):
        imputer = KNNImputer(n_neighbors= n_neighbors)
        # imputer = KNNImputer(n_neighbors= n_neighbors)
        data_i = data[slice_data[i],:]
        imputer.fit(data_i)
        model_path = directory_save_model + "KNN_imputer_slice_{}.pkl".format(str(i))
        joblib.dump(imputer, model_path)
        list_model_imputer.append(model_path)
    return list_model_imputer

def Iterative_Imputer(data, n_step, max_iter, directory_save_model):
    """
    Process missing data by Iterative Imputer
    Args:
        data: Data missing
        n_step: slice data to n_step part
        max_iter: max_iter in Iterative
        directory_save_model: dir save model Iterative_Imputer
    Return:
        list file dir path of model Imputer
    """
    len_data = data.shape[0]
    slice_data = np.arange(len_data)
    slice_data = np.array(np.split(slice_data, len_data//n_step))
    slice_data = np.reshape(slice_data, len_data, order='F')
    slice_data = np.split(slice_data, n_step)
    list_model_imputer = []
    if not os.path.exists(directory_save_model):
        os.makedirs(directory_save_model)
    for i in range(len(slice_data)):
        imputer = IterativeImputer(max_iter= max_iter, random_state=0)
        # imputer = IterativeImputer(max_iter= max_iter, random_state=0)
        data_i = data[slice_data[i],:]
        imputer.fit(data_i)
        model_path = directory_save_model + "Iterative_imputer_slice_{}.pkl".format(str(i))
        joblib.dump(imputer, model_path)
        list_model_imputer.append(model_path)
    return list_model_imputer

def XGB_Imputer(data, n_step, max_iter, directory_save_model):
    """
    Process missing data by XGB Imputer
    Args:
        data: Data missing
        n_step: slice data to n_step part
        max_iter: max_iter in Iterative
        directory_save_model: dir save model XGB_Imputer
    Return:
        list file dir path of model Imputer
    """
    len_data = data.shape[0]
    slice_data = np.arange(len_data)
    slice_data = np.array(np.split(slice_data, len_data//n_step))
    slice_data = np.reshape(slice_data, len_data, order='F')
    slice_data = np.split(slice_data, n_step)
    list_model_imputer = []
    if not os.path.exists(directory_save_model):
        os.makedirs(directory_save_model)
    for i in range(len(slice_data)):
        imputer = IterativeImputer(estimator=XGBRegressor(), 
                    max_iter= max_iter, random_state=0,
                    initial_strategy='median',skip_complete=True)
        # imputer = IterativeImputer(estimator=XGBRegressor(), max_iter= max_iter, random_state=0, initial_strategy='median',skip_complete=True)
        data_i = data[slice_data[i],:]
        imputer.fit(data_i)
        model_path = directory_save_model + "XGB_imputer_slice_{}.pkl".format(str(i))
        joblib.dump(imputer, model_path)
        list_model_imputer.append(model_path)
    return list_model_imputer

def fill_missing_knn(data,n_step, list_model_imputer):
    # Fill missing data with Imputer model
    len_data = data.shape[0]
    slice_data = np.arange(len_data)
    slice_data = np.array(np.split(slice_data, len_data//n_step))
    slice_data = np.reshape(slice_data, len_data, order='F')
    slice_data = np.split(slice_data, n_step)
    data_fill = np.zeros_like(data)
    for i in range(len(slice_data)):
        imputer_model = joblib.load(list_model_imputer[i])   
        data_fill[slice_data[i],:] = imputer_model.transform(data[slice_data[i],:])
    return data_fill