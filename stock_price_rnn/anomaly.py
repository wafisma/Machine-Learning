import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.metrics import mean_squared_error
import time

import lstm
import utils.visualize as vs
import utils.stock_data as sd

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

#model = lstm.build_model_lautoencoder(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)


model = Sequential()
model.add(Bidirectional(LSTM( input_shape=(None,X_train.shape[-1]),
        units=unroll_length, return_sequences=True),
                       ))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(100)))
model.add(Dropout(0.2))

model.add(Dense(
        units=1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='adam')


start = time.time()
#Training the model with train data for 10 epochs, see docs for model.fit 
model.fit(X_train,y_train,batch_size=32,epochs=5,validation_split=0.1)
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
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

#Visualize
#To see visualization, use utils.visualize.plot_lstm_prediction function. Plots y_test vs predictions. Use save=True if you want to save the image directly.
vs.plot_lstm_prediction(y_test, predictions, title='GSPC_predictions', y_label='Price USD', x_label='Trading Days', save=True)


