import tensorflow as tf
keras = tf.keras
import numpy as np

def create_tf_data_2_in_1_out(
    X,
    Y,
    sequence_stride,
    sequence_length_x,
    sequence_length_y,
    sampling_rate=4,
    batch_size = 32,
    shuffle=True
    ):
    """
    Create dataset tensorflow for model
    Args:
        X: Data inputs
        Y: Data outputs
        sequence_stride: example sequence_stride =2, [1,2,3,4] --> [3,4,5,6]
        sequence_length_x: len of sequence inputs
        sequence_length_y: len of sequence outputs
        sampling_rate: split squence to sampling, example: sequence days to 24 sequence hours
    Return:
        tf dataset 
    """
    list_x1 = list(X.columns[:-10])
    list_x2 = list(X.columns[-10:])
    inputs1 = keras.preprocessing.timeseries_dataset_from_array(
        X.iloc[:, list_x1],
        None,
        sequence_stride = sequence_stride,
        sequence_length= sequence_length_x,
        sampling_rate= sampling_rate,
        batch_size=1,
    )
    inputs1 = np.stack(list(inputs1))
    inputs1 = np.squeeze(inputs1, axis=1)

    inputs2 = keras.preprocessing.timeseries_dataset_from_array(
        X.iloc[:, list_x2],
        None,
        sequence_stride = sequence_stride,
        sequence_length= sequence_length_x //(sampling_rate * 3),
        sampling_rate=sampling_rate * 3,
        batch_size=1,
    )
    inputs2 = np.stack(list(inputs2))
    inputs2 = np.squeeze(inputs2, axis=1)

    outputs = keras.preprocessing.timeseries_dataset_from_array(
        Y,
        None,
        sequence_stride = sequence_stride,
        sequence_length= sequence_length_y,
        sampling_rate=sampling_rate,
        batch_size=1,
    )
    outputs = np.stack(list(outputs))
    outputs = np.squeeze(outputs, axis=1)

    dataset_in =  tf.data.Dataset.from_tensor_slices((inputs1, inputs2))
    dataset_out = tf.data.Dataset.from_tensor_slices(outputs)
    dataset = tf.data.Dataset.zip((dataset_in, dataset_out)).batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset