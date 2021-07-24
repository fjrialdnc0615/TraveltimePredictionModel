# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 00:31:25 2021

@author: a1381
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


tf.compat.v1.random.set_random_seed(1234)

file = 'from01F0147Sto01F0155S.csv'
df = pd.read_csv(file)
df = df[['speed','two_station_volumes','volume_x','volume_y','traveltime']]

# Split train data and test data
train_size = int(len(df)*0.8)

# train_data = df.WC.loc[:train_size] -----> it gives a series
# Do not forget use iloc to select a number of rows
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]




# Scale data
# The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
scaler = MinMaxScaler().fit(train_data)

train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

def slidingwindow(data , sliding_windows):
    #取'speed','two_station_volumes','volume_x','volume_y'
    X = data
    
    #取'traveltime'
    y = data[:,4]
    
    #放feature
    merge_array = []
    
    #放label
    label = []
    
    #總共移動次數
    moving_distance = len(X) - sliding_windows
    
    
    for i in range(moving_distance):
        
        #取i~i+sliding windows長度的sample ex 0~11筆
        window = X[i:i + sliding_windows]
        #丟入一串列
        merge_array.append(window)    
        
        #取i+sliding windows長度的sample ex 第12筆
        label_pd = y[sliding_windows + i]
        label.append([label_pd])    
    #轉為numpy
    merge_array = np.array(merge_array)
    label = np.array(label)
    
    return merge_array, label

length_of_slidingwindows = 12
X_train, y_train = slidingwindow(train_scaled,length_of_slidingwindows)
X_test, y_test = slidingwindow(test_scaled,length_of_slidingwindows)
    
    
def create_gru(units):
    model = Sequential()
    # Input layer 
    model.add(GRU (units = units, return_sequences = True, 
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2)) 
    # Hidden layer
    model.add(GRU(units = units,return_sequences=True))
    model.add(GRU(units = units))            
    model.add(Dropout(0.2))
    model.add(Dense(units = 1)) 
 
    #Compile model
    model.compile(optimizer='adam',loss='mse')
   
    return model

model_gru = create_gru(64)
y_pred_gru = []

def fit_model(model):

    global test_data
    global y_pred_gru
    global length_of_slidingwindows
    
    turning = []

    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)
    history = model.fit(X_train, y_train, epochs = 50, validation_split = 0.2,
                    batch_size = 8192, shuffle = False, callbacks = [early_stop])
    print('finish')
    #得到等等要預測的label進而去計算residual
    test_label = test_data[['traveltime']]
    test_label = test_label.values.tolist()
    test_label = np.array(test_label)
    test_label = test_label.flatten()
    test_label = test_label.tolist()
    #因為不是slidingwindows，所以會多n筆，所以把前n筆去掉
    test_label = test_label[length_of_slidingwindows:]

    #record 是 predict的label
    record = model.predict(X_test)
    record = record.flatten()
    
    #turning存放predict label
    for i in record:
        turning.append(i * (691 - 21) + 21)
    y_pred_gru = np.array(turning)
    
    print('gru_mae:',mean_absolute_error(test_label, turning))
    print('gru_rmse:',mean_squared_error(test_label, turning)**0.5)
    print('gru_Acc:', 1 - mean_absolute_percentage_error(test_label, turning))
    
    return history

history_gru = fit_model(model_gru)
print('-------------------------------------------')








###XG_boost







file = 'from01F0147Sto01F0155S.csv'
df2 = pd.read_csv(file)
df2 = df2[['traveltime']]

train_size = int(len(df2)*0.8)

# train_data = df.WC.loc[:train_size] -----> it gives a series
# Do not forget use iloc to select a number of rows
train_data = df2.iloc[:train_size]
test_data = df2.iloc[train_size:]

train_data = train_data.values.flatten()
test_data = test_data.values.flatten()

train_data = train_data.tolist()
test_data = test_data.tolist()

def sliding_windows(length,data):
    X = []
    Y = []
    moving_distance = len(data) - length
    for i in range(moving_distance):
        window = data[i:i+length]
        X.append(window)
        
        label = data[i+length]
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y) 
    return X, Y
X_train, y_train = sliding_windows(12, train_data)
X_test, y_test = sliding_windows(12, test_data)
# split data into train and test sets
seed = 7
test_size = 0.2

# fit model no training data
model = XGBRegressor()
model.fit(X_train, y_train)

# make predictions for test data
y_pred_xgboost = model.predict(X_test)
print('xgboost_mae:',mean_absolute_error(y_test, y_pred_xgboost))
print('xgboost_rmse:',mean_squared_error(y_test, y_pred_xgboost)**0.5)
print('xgboost_Acc:', 1 - mean_absolute_percentage_error(y_test, y_pred_xgboost))
print('-------------------------------------------')
y_input_linear = np.stack((y_pred_gru, y_pred_xgboost), axis=1)





#hybrid model




model = LinearRegression()
model.fit(y_input_linear, y_test)
y_pred = model.predict(y_input_linear)
print('hybrid_mae:',mean_absolute_error(y_test, y_pred))
print('hybrid_rmse:',mean_squared_error(y_test, y_pred)**0.5)
print('hybrid_Acc:', 1 - mean_absolute_percentage_error(y_test, y_pred))
#y_input_linear = np.concatenate((y_pred_xgboost, y_pred_gru), axis=1)
#print(y_input_linear)
# evaluate predictions
# evaluate predictions
print('-------------------------------------------')