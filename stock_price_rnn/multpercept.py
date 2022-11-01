# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:37:30 2020

@author: sir wafaa
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#this is the imput vector\features
x= np.array([[0.0,0.0],[0.2,0.3],[0.45,0.2]])
#y targets values/labels
y= np.array([[0.0],[0.5],[0.65]])

model= Sequential()

model.add(Dense(50, input_dim=2, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])
print(model.summary())

input()

model.fit(x,y, epochs=3000)

#non linear dataset
X= np.array([[0.3,0.3],[0.4,0.5],[1.0,0.0]])
Y= np.array([[0.6],[0.9],[1.0]])

z= model.predict(x)

for i,j in zip(Y,z):
    print('{} => {}'.format(i,j))