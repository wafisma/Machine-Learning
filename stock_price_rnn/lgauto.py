# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:26:22 2020

@author: sir wafaa
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn import metrics
from sklearn.metrics import mean_squared_error
# %matplotlib inline
# config InlineBackend.figure_format='retina'

# register_matplotlib_converters()
# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 22, 10

# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')

train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['close']])

train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train[['close']], train.close, TIME_STEPS)
X_test, y_test = create_dataset(test[['close']], test.close, TIME_STEPS)

print(X_train.shape)


model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64, 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.GRU(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(
    units=X_train.shape[2]
)))

model.compile(loss='mse', optimizer='adam')


history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    shuffle=False
)

#history =model.fit(X_train,y_train,batch_size=32,epochs=3,validation_split=0.1)
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

print('Train loss: MAE', np.mean(train_mae_loss))
print('Test loss:  MAE', np.mean(test_mae_loss))
#print('Difference in loss:  MAE', np.mean(test_mae_loss-train_mae_loss))
# lstm_train_pred = model.predict(X_train)
# lstm_val_pred = model.predict(X_test)
# print('Train rmse:', np.sqrt(mean_squared_error(y_train, lstm_train_pred)))
# print('Test rmse:', np.sqrt(mean_squared_error(y_test, lstm_val_pred)))

# trainScore = model.evaluate(X_train, X_train, verbose=0)
# print('Train Score: %.8f MAE' % (trainScore))

# testScore = model.evaluate(X_test, X_test, verbose=0)
# print('Test Score: %.8f MAE' % (testScore))

# print("Train set", X_train_encoded.shape)
# print("Validation set", X_valid_encoded.shape)
#print ('\nTest Scores: mse={}, mae={}, mape={}'.format(*testScore))
# print("Mean Absolute Error :",metrics.mean_absolute_error(X_test, X_test))
# print("Mean Squared Error:",metrics.mean_squared_error(X_test, X_test))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();



# print(train_mae_loss)
# print(test_mae_loss)

#print ('\nTest Scores:  loss={}, val_loss={}'history.history.keys())
# list all data in history
#print(history.history.keys())
#tes plot
# plt.plot(y_train, color = 'black', label = 'TATA Stock Price')
# plt.plot(y_test, color = 'green', label = 'Predicted TATA Stock Price')
# plt.title('TATA Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('TATA Stock Price')
# plt.legend()
# plt.show()