# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 08:39:40 2020

@author: sir wafaa
"""

from tensorflow.keras.metrics import *
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import math




import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
   
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.array([112.3,108.4,148.9,117.4])
    y_pred = np.array([124.7,103.7,116.6,78.5])
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_percentage_error(112.3, 124.7))
# def mean_absolute_percentage_error(y_true, y_pred):
#     """
#     Calculate mean absolute percentage error, ignoring the magic number

#     :param y_true: Ground Truth
#     :type y_true: Union(tf.Tensor, tf.Variable)
#     :param y_pred: Prediction
#     :type y_pred: Union(tf.Tensor, tf.Variable)
#     :return: Mean Absolute Percentage Error
#     :rtype: tf.Tensor
#     :History: 2018-Feb-17 - Written - Henry Leung (University of Toronto)
#     """
#     tf_inf = tf.cast(tf.constant(1) / tf.constant(0), tf.float32)
#     epsilon_tensor = tf.cast(tf.constant(tfk.backend.epsilon()), tf.float32)

#     diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), epsilon_tensor, tf_inf))
#     diff_corrected = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), diff)
#     return 100. * tf.reduce_mean(diff_corrected, axis=-1) * magic_correction_term(y_true)
# print(mean_absolute_percentage_error(23, 25))
    


# def greet(name):
#     """
#     This function greets to
#     the person passed in as
#     a parameter
#     """
#     print("Hello, " + name + ". Good morning!")
  
# greet('Paul')
    
# def mean_absolute_percentage_error(y_true, y_pred):
   
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# print(mean_absolute_percentage_error(112.3, 124.7))


# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# y_true = np.random.randn(100)
# y_pred = y_true * 3.5

# print(mean_absolute_percentage_error(y_true, y_pred))


# from sklearn.utils import check_arrays
# import numpy as np
# def mean_absolute_percentage_error(y_test, predictions): 
#     """
#     Use of this metric is not recommended; for illustration only. 
#     See other regression metrics on sklearn docs:
#       http://scikit-learn.org/stable/modules/classes.html#regression-metrics
#     Use like any other metric
#     >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     Out[]: 24.791666666666668
#     """

#     y_test, predictions = check_arrays(y_test, predictions)

#     ## Note: does not handle mix 1d representation
#     #if _is_1d(y_true): 
#     #    y_true, y_pred = _check_1d_array(y_true, y_pred)

#     return np.mean(np.abs((y_test - predictions) / y_test)) * 100