# implementation of a simple autoencoder for anomaly detection on time series
# see : https://www.tensorflow.org/tutorials/generative/autoencoder#third_example_anomaly_detection

# How will you detect anomalies using an autoencoder? 
# Recall that an autoencoder is trained to minimize reconstruction error. 
# You will train an autoencoder on the normal series only, then use it to reconstruct all the data. 
# Our hypothesis is that the abnormal series will have higher reconstruction error. 
# You will then classify a series as an anomaly if the reconstruction error surpasses a fixed threshold.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tensorflow.keras.models import Model

# load here a dataframe named df
df.head()

#data shape (pandas object):
#		F1	F2	F3	F4
#	1   27   4  10   7
#	2    9  10  20   1
#	3   13  29   8   2
#	4   19   9  21   3
#	5   29   2  20   4
#	6   41   6  29   3


# make a robust normalization (robust to anomalies)
min_val = tf.reduce_min(df_sc)
max_val = tf.reduce_max(df_sc)

df_sc = (df_sc - min_val) / (max_val - min_val)

df_sc = tf.cast(df_sc, tf.float32)

# no train-test separation since no label
# only some (possibly) normal data

# separate the normal series from the others
data_c = #clean data
data_o = #other data

# built the model
class AnomalyDetector(Model):
	def __init__(self):
    	super(AnomalyDetector, self).__init__()
    	self.encoder = tf.keras.Sequential([
      		layers.Dense(32, activation="relu"),
      		layers.Dense(16, activation="relu"),
      		layers.Dense(8, activation="relu")
      		])

    	self.decoder = tf.keras.Sequential([
      		layers.Dense(16, activation="relu"),
      		layers.Dense(32, activation="relu"),
      		layers.Dense(140, activation="sigmoid")
      		])  #should adapt the output format

  	def call(self, x):
    	encoded = self.encoder(x)
    	decoded = self.decoder(encoded)
    	return decoded

autoencoder = AnomalyDetector()


autoencoder.compile(optimizer='adam', loss='mae')

# autoencoder is trained with only normal series

history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
# to adjust ! no test data, only "other" data ()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


# Detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold.





# https://blog.tensorflow.org/2020/04/how-airbus-detects-anomalies-iss-telemetry-data-tfx.html
# https://arxiv.org/pdf/1607.00148.pdf
# The objective of the model is to represent the nominal state of the subsystem. 
# If the model is able to reconstruct observations of nominal states with a high accuracy, 
# it will have difficulties reconstructing observations of states which deviate from the nominal state. 
# Thus, the reconstruction error of the model is used as an indicator for anomalies during inference,
# as well as part of the cost function in training. 

# The Autoencoder uses LSTMs to process sequences and capture temporal information. 
# Each observation is represented as a tensor with shape [number_of_features,number_of_timesteps_per_sequence].
# The data is prepared using TFT’s scale_to_0_1 and vocabulary functions. 
# Each LSTM layer of the encoder is followed by an instance of tf.keras.layers.Dropout to increase the robustness against noise. 

# so we need the number of features and the number of measures for each series
# 32 -> 16 -> 8 -> 16 -> 32 suffices

# see also : https://blog.keras.io/building-autoencoders-in-keras.html section sequence-to-sequence autoencoder