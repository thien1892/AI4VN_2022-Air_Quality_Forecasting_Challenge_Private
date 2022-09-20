import pandas as pd
from geopy.distance import lonlat, distance
import numpy as np
import glob
import os

def dict_long_lat(path_file_csv):
    """
    Load location long, lat from file csv
    Args:
        path_file_csv: path of file csv
    Returns:
        dict with keys are name of location and values are long, lat of location
    """
    df = pd.read_csv(path_file_csv)
    if 'location' in df.columns:
        df = df[['location', 'longitude', 'latitude']]
    elif 'stat_name' in df.columns:
        df.columns = ['location','latitude', 'longitude']
    long_lat = [(df.longitude.to_list()[i], df.latitude.to_list()[i]) for i in range(df.shape[0])]
    return dict(zip(df.location.to_list(), long_lat))

def dis_km_2_station(s1,s2):
    """
    Calculate distance (km) of two location
    """
    return distance(lonlat(*s1), lonlat(*s2)).km

def dis_km_3_station_mean(s1,s2,s3):
    """
    Calculate mean distance (s1,s3) and (s2,s3)
    """
    dis_s1_s3 = distance(lonlat(*s1), lonlat(*s3)).km
    dis_s2_s3 = distance(lonlat(*s2), lonlat(*s3)).km
    return (dis_s1_s3 +dis_s2_s3 )/ 2.

def dis_km_3_station_max(s1,s2,s3):
    """
    Calculate max distance (s1,s3) and (s2,s3)
    """
    dis_s1_s3 = distance(lonlat(*s1), lonlat(*s3)).km
    dis_s2_s3 = distance(lonlat(*s2), lonlat(*s3)).km
    return max(dis_s1_s3 ,dis_s2_s3 )

def dis_two_file_location(file_csv1, file_csv2, type_dis = 'max'):
    '''
    file1: location output
    file2: locatiom input
    Return len(file2) * len(file2), each component is dis station(file2) --> station(out) --> station(file2)
    '''
    dict_distance = {}
    location1 = dict_long_lat(file_csv1)
    location2 = dict_long_lat(file_csv2)
    for k3,v3 in location1.items():
        data_distance = []
        for _, v1 in location2.items():
            for _ ,v2 in location2.items():
                if type_dis == 'max':
                    data_distance.append(dis_km_3_station_max(v1,v2,v3))
                if type_dis == 'mean':
                    data_distance.append(dis_km_3_station_mean(v1,v2,v3))
        np_data_distance = np.array(data_distance)
        np_data_distance = np_data_distance.reshape((len(location2),len(location2)))
        dict_distance[k3] = np_data_distance
    return dict_distance

def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

def compute_adjacency_matrix_custom(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    num_routes_out = route_distances.shape[0]
    num_routes_in = route_distances.shape[1]
    route_distances = route_distances / 10.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes_out, num_routes_in])
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

def get_location_test(path_shape_file):
    """
    input: './data_test/input/*/location_input.csv'
    output: './data_test/input/*/location_output.csv'
    """
    list_file_test_input = glob.glob(path_shape_file)
    df_test_location_input = pd.DataFrame()
    for f in list_file_test_input:
        path = os.path.normpath(f)
        name_folder = path.split(os.sep)[-2]
        df = pd.read_csv(f)
        df['folder'] = name_folder
        if len(df_test_location_input.columns) < 3:
            df_test_location_input.columns = pd.DataFrame(columns= df.columns)
        df_test_location_input = pd.concat((df_test_location_input, df))

    df_test_location_input = df_test_location_input.drop_duplicates(subset=['longitude', 'latitude'])
    if 'output' in path_shape_file:
        stt_location = ['location_test_'+ str(i) for i in range(len(df_test_location_input))]
        df_test_location_input['location'] = stt_location
    return df_test_location_input

def check_nearest_location(file_csv1, file_csv2, dis_near = 5):
    '''
    file1: location output
    file2: locatiom input
    '''
    dict_location = {}
    location1 = dict_long_lat(file_csv1)
    location2 = dict_long_lat(file_csv2)
    for k1,v1 in location1.items():
        data_distance = {}
        for k2, v2 in location2.items():
            if dis_km_2_station(v1,v2) < dis_near:
                data_distance[k2] = dis_km_2_station(v1,v2)
        # np_data_distance = np.array(data_distance)
        dict_location[k1] = data_distance
    return dict_location

def check_n_nearest_location(file_csv1, file_csv2, n_near = 3):
    '''
    file1: location output
    file2: locatiom input
    '''
    dict_location = {}
    location1 = dict_long_lat(file_csv1)
    location2 = dict_long_lat(file_csv2)
    for k1,v1 in location1.items():
        data_distance = {}
        data_distance_n = {}
        for k2, v2 in location2.items():
            data_distance[k2] = dis_km_2_station(v1,v2)
        data_distance = dict(sorted(data_distance.items(), key=lambda item: item[1]))
        i = 0
        for k,v in data_distance.items():
            if i < n_near:
                data_distance_n[k] = v
            i += 1

        # np_data_distance = np.array(data_distance)
        dict_location[k1] = data_distance_n
    return dict_location