# -*- coding: utf-8 -*-
"""
LSTM for APPLE

@author: TSE
Modified by: danielroa
"""
import numpy as np
import pandas as pd

df=pd.read_csv('AAPL.csv')

# Extracting the close price from the original dataframe
df1=df['close']

# Plotting the closing price of the Apple Stock
import matplotlib.pyplot as plt
plt.plot(df1)

# Scaling the values
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
# Using 65% of the data for training
training_size=int(len(df1)*0.65)
# The rest of the data is for testing
test_size=len(df1)-training_size

# Seperating the data
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
	dataX, dataY = [], []
                   
	for i in range(len(dataset)-time_step-1):            
		a = dataset[i:(i+100), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
# We convert the data from columns to rows
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
# The 1 at the end represents the amount of features you are capturing
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()                                  #timeStep, [close]
# Creation and implementation of the model
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
# We attach another layer, by setting the return sequence to true we are carrying the memory from previous layers-
# The longer the time step, the bigger the sequence
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
# We add a regular output layer
model.add(Dense(1))
# Since its a regressor, we set the compile with MSE and ADAM
model.compile(loss='mean_squared_error',optimizer='adam')

# =============================================================================
# Execution of the epochs
# =============================================================================
# model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100,
#          batch_size=64, verbose=1)
# =============================================================================
# Save Model Using Keras
# This saves the model and preserves the learning from it
# =============================================================================
# model.save("LSTMApple.h5")

# =============================================================================
# Use the Saved Model Using Keras
# =============================================================================
# Once its loaded, all the weights are now loaded 
model= keras.models.load_model('LSTMApple.h5')
# =============================================================================

### Lets Do the prediction and check performance metrics
# The output of the variables are scaled values
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

## Inverse transform to scale it back to their original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
y_trainUnScaled = scaler.inverse_transform(y_train.reshape(-1,1))
y_testUnScaled =  scaler.inverse_transform(ytest.reshape(-1,1))

import math
from sklearn.metrics import mean_squared_error
#Trainerror = math.sqrt(mean_squared_error(y_train,train_predict))
TrainerrorNew = math.sqrt(mean_squared_error(y_trainUnScaled,train_predict))

### Test Data RMSE
#TestError = math.sqrt(mean_squared_error(ytest,test_predict))
TestErrorNew = math.sqrt(mean_squared_error(y_testUnScaled,test_predict))

### Plotting 
look_back=100

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[training_size+look_back:len(df1)-1, :] = test_predict

# plot baseline and predictions
# Yellow -> real trend
# Blue -> predicted trend
plt.plot(scaler.inverse_transform(df1),c="yellow")
plt.plot(testPredictPlot)
plt.show()










