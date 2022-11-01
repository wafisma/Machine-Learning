# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:17:42 2020

@author: sir wafaa
"""

import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
import time
import tensorflow as tf
import lstm
import utils.visualize as vs
import utils.stock_data as sd
from sklearn import metrics

# Hossain, M. A., Karim, R., Thulasiram, R., Bruce, N. D. B., & Wang, Y. (2018). Hybrid Deep
# Learning Model for Stock Price Prediction. Proceedings of the 2018 IEEE Symposium 
# Series on Computational Intelligence, SSCI 2018, 1837–1844.
# https://doi.org/10.1109/SSCI.2018.8628641


# dataloading
stocks = pd.read_csv('GSPC_preprocessed.csv')

#training on LSTM
stocks_data = stocks.drop(['Item'], axis =1)

print (stocks_data.shape[0])
test_data_sz=int(stocks_data.shape[0]*0.2)
print (test_data_sz)

X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5,test_data_size=test_data_sz) # 5 outputs
'''
X_train: input data for training
y_train: ground truth for training
X_test: input data for testing
y_test: ground truth for testing
'''

print ('%d %d'  %(len(X_train),len(X_test)))
unroll_length = 50
validation_split=0.05

X_train = sd.unroll(X_train, unroll_length)
X_test = sd.unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]

#print X_train.shape[-1],y_train.shape
#exit()

#Building model. to change model architecture, can use any of other two functions in lstm.py or define new model 
model = lstm.build_model_bilbigrumlp(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mape'])


start = time.time()
#Training the model with train data for 10 epochs, see docs for model.fit 
model.fit(X_train,y_train,batch_size=32,epochs=1,validation_split=0.1)
t0=time.time()

#Generating predictions for test data X_test
#X_test: test input 
# y_test: test ground truth
#predictions: model predictions for test data
predictions = model.predict(X_test) 
print ('prediction size: %s '%str(predictions.shape))
t1=time.time()

print( 'test time: '+str(t1-t0))
testScore = model.evaluate(X_test, y_test, verbose=0)
#print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
print ('\nTest Scores:  mse={}, mae={}, mape={}'.format(*testScore))
# print("Mean Absolute Error :",metrics.mean_absolute_error(y_test, predictions))
# print("Mean Squared Error:",metrics.mean_squared_error(y_test, predictions))
# def mean_absolute_percentage_error(y_true, y_pred):
   
#      y_true, y_pred = np.array(y_true), np.array(y_pred)
    
#      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# print("Mean Squared Percentage Error:",mean_absolute_percentage_error(y_test, predictions))

#print("Mean Absolute Percentage Error:",metrics.mean_absolute_percentage_error(y_test, predictions))

#print("MeanAbsolutePercentageError:",tf.keras.metrics.MeanAbsolutePercentageError(y_test, predictions))
#print("MeanAbsolutePercentageError:",metrics.mean_absolute_percentage_error(y_test, predictions))
# def mean_absolute_percentage_error(y_test, predictions): 
#   y_test, predictions = np.array(y_test), np.array(predictions)
  
  # y_test = np.random.randn(100)
  # predictions = y_test * 3.5
  #if _is_1d(y_true): 
  #    y_true, y_pred = _check_1d_array(y_true, y_pred)
  # return  np.mean(np.abs((y_test - predictions) / y_test)) * 100

#print("Mean Absolute Percentage Error:",metrics.mean_absolute_percentage_error(y_test, predictions))

#Visualize
#To see visualization, use utils.visualize.plot_lstm_prediction function. Plots y_test vs predictions. Use save=True if you want to save the image directly.
vs.plot_lstm_prediction(y_test, predictions, title='GSPC_predictions', y_label='Price USD', x_label='Trading Days', save=True)


# print("##############################################################")
# print("Accuracy:",metrics.accuracy_score(y_test, predictions))
# print("Kappa Stats:",metrics.cohen_kappa_score(y_test, predictions))
# print("Precision:",metrics.precision_score(y_test, predictions))
# print("Recall:",metrics.recall_score(y_test, predictions))
# print("Mean Absolute Error:",metrics.mean_absolute_error(y_test, predictions))
# print("Mean Squared Error:",metrics.mean_squared_error(y_test, predictions))
# print("F-Measure:",metrics.recall_score(y_test, predictions))
# print("##############################################################")
#mean_absolute_percentage_error(y_test, X_test)
#Mloss=100 * abs(y_test - predictions) / y_test
# Mloss = 100 * abs(y_test - X_test) / y_test
#print('Test Score mloss: %.8f MAPE ' % (Mloss))
#print(testMape)
#print ('\nTest Scores: mape={}'.format(*testScore))
#print('MAPE: %.8f MAPE'.format(Mloss))
#return ( abs((y_test - X_test) / y_test).mean()) * 100
#print ('My MAPE: ' + str(MAPE(X_test,y_test)) )
#Mloss= abs((y_test - X_test) / y_test).mean() * 100
#print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
#print('manual result: mse=%8f' mean(math.square(y_test - X_test)))
#print(testScore)+ '{%.8f}'


