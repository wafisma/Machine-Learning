# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 08:22:53 2020

@author: sir wafaa
"""

# lstm autoencoder predict sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dense, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
#from keras.utils import plot_model
# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define model
model = Sequential()
model.add(LSTM(100, input_shape=(n_in,1)))
model.add(Dropout(0.2))
model.add(RepeatVector(n_out))
model.add(GRU(100,activation='linear',return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

#plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
# fit model
model.fit(seq_in, seq_out, epochs=150, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat[0,:,0])