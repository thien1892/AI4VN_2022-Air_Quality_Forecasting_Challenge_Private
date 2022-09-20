import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Input
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Activation
from tensorflow.keras import Model

def model_2_in_1_out(
    input1_shape_0: int,
    input1_shape_1: int,
    input2_shape_0: int,
    input2_shape_1: int,
    output_shape_0: int,
    output_shape_1: int,
    drop_out: float,
    n_LSTM: int,
    n_Dense: int,
    ):
    input1 = Input(shape=(input1_shape_0, input1_shape_1))
    M1 = LSTM(n_LSTM)(input1)
    M1 = BatchNormalization()(M1)
    M1 = Activation('relu')(M1)
    M1 = Dropout(drop_out)(M1)
    output1 = Dense(n_Dense)(M1)

    input2 = Input(shape=(input2_shape_0, input2_shape_1))
    M2 = LSTM(n_LSTM//2)(input2)
    M2 = BatchNormalization()(M2)
    M2 = Activation('relu')(M2)
    M2 = Dropout(drop_out)(M2)
    output2 = Dense(n_Dense//4)(M2)

    input3 = keras.layers.Concatenate()([output1, output2])
    M = BatchNormalization()(input3)
    M = Activation('relu')(M)
    M = Dense(n_Dense * 2 // 3)(M)
    M = RepeatVector(output_shape_0)(M)
    M = BatchNormalization()(M)
    M = Activation('relu')(M)
    M = Dropout(drop_out)(M)
    output = TimeDistributed(Dense(output_shape_1))(M)
    
    model = Model(inputs = [input1, input2], outputs = output)
    return model

def model_2_in_1_out_2(
    input1_shape_0: int,
    input1_shape_1: int,
    input2_shape_0: int,
    input2_shape_1: int,
    output_shape_0: int,
    output_shape_1: int,
    drop_out: float,
    n_LSTM: int,
    n_Dense: int,
    ):
    input1 = Input(shape=(input1_shape_0, input1_shape_1))
    M1 = LSTM(n_LSTM)(input1)
    M1 = BatchNormalization()(M1)
    M1 = Activation('relu')(M1)
    M1 = Dropout(drop_out)(M1)
    output1 = Dense(n_Dense)(M1)

    input2 = Input(shape=(input2_shape_0, input2_shape_1))
    M2 = LSTM(n_LSTM//2)(input2)
    M2 = BatchNormalization()(M2)
    M2 = Activation('relu')(M2)
    M2 = Dropout(drop_out)(M2)
    output2 = Dense(n_Dense//4)(M2)

    input3 = keras.layers.Concatenate()([output1, output2])
    M = BatchNormalization()(input3)
    M = Activation('relu')(M)
    M = Dense(n_Dense * 2 // 3)(M)
    M = RepeatVector(output_shape_0)(M)
    M = BatchNormalization()(M)
    M = Activation('relu')(M)
    M = Dropout(drop_out)(M)
    M = LSTM(n_LSTM, return_sequences=True)(M)
    M = BatchNormalization()(M)
    M = Activation('relu')(M)
    M = Dropout(drop_out)(M)

    output = TimeDistributed(Dense(output_shape_1))(M)
    
    model = Model(inputs = [input1, input2], outputs = output)
    return model