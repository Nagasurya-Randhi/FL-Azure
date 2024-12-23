import numpy as np
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, GRU, Activation, Flatten, RepeatVector, Permute, multiply, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from pykalman import KalmanFilter

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

# Define the attention-based GRU model
def create_attention_model_GRU(input_shape, units):
    inputs = Input(shape=input_shape)
    cnn_out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = Dropout(0.2)(cnn_out)
    
    gru_out = GRU(units, return_sequences=True)(cnn_out)
    
    attention = Dense(1, activation='relu')(gru_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([gru_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    output = Dense(1)(sent_representation)
    model = Model(inputs=inputs, outputs=output)
    return model

# Function to create dataset for time series forecasting
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
