import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer

from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# get data as pandas DataFrame
dataset = pd.read_csv("admissions_data.csv")

# print(dataset.head())

# split data into features and labels
# no categorical variables, hence no need to map them to numerical values

# all columns apart from first (serial number) and last (results) are features
features = dataset.iloc[:, 1:-1]
# las column has the results
labels = dataset.iloc[:, -1]

# split into training set and testing set
features_train,  features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 21)

# scale data

# initialize scaling obj
ct = ColumnTransformer([("scale", StandardScaler(), features.columns)], remainder = 'passthrough')
# fit ColumnTransformer obj to the training data and transform it
features_train_scaled = ct.fit_transform(features_train)
# transform test features
features_test_scaled = ct.transform(features_test)

# function to create model
def create_model(features, learning_rate):
  model = Sequential()
  # create & add input layer
  model.add(InputLayer(input_shape = (features.shape[1], )))
  # add hidden layer with a dropout layer
  model.add(Dense(32, activation = "relu"))
  model.add(layers.Dropout(0.2))
  # add output layer
  model.add(Dense(1))

  # initialise Adam optimizer
  optimizer = Adam(learning_rate = learning_rate)

  model.compile(loss = "mse", metrics = ["mae"], optimizer = optimizer)
  return model

# hyperparameters
learning_rate = 0.01
epochs = 40
batch_size = 1

# initialise model
model = design_model(features_train_scaled, learning_rate)

# callback function to monitor the validation loss
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# fit the model
history = model.fit(features_train_scaled, labels_train, epochs = epochs, batch_size = batch_size, verbose = 0, validation_split=0.2, callbacks=[stop])
# evaluate the model
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose = 0)

print('MSE: ', res_mse)
print('MAE: ', res_mae)

# evaluate R-squared value
predicted_values = model.predict(features_test_scaled)
print(r2_score(labels_test, predicted_values))

#plot the learning curve
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping each other
fig.tight_layout()
fig.savefig('static/images/my_plots.png')
