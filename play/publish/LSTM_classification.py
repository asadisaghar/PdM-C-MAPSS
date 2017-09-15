import sys
import os.path
import itertools
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.metrics
import keras.optimizers
import keras.callbacks
import keras.utils.np_utils
import keras.preprocessing.sequence
from keras.objectives import categorical_crossentropy
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import warnings
warnings.filterwarnings('ignore')
import LSTM_tools as tools
from keras.utils.np_utils import to_categorical

## Settings
merge_data = True
sn = 3
usecols = None
setnumber = 'FD00' + str(sn)
data_path = 'data'
plot_path = 'plots'
model_path = 'models'

## MLP settings
dense_width = 8
dense_depth = 1
dropout = 0.5
bs = 12 #batch size
epochs = 50

train, test, validate = tools.read_preprocessed_data(setnumber, usecols, merge_data)
agg = tools.series_to_supervised(train, n_in=1, n_out=1)
X_train = agg[agg.columns.difference(['var16(t)'])].values
y_train = to_categorical(agg['var16(t)'].values)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

agg = tools.series_to_supervised(validate, n_in=1, n_out=1)
X_validate = agg[agg.columns.difference(['var16(t)'])].values
y_validate = to_categorical(agg['var16(t)'].values)
X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))

agg = tools.series_to_supervised(test, n_in=1, n_out=1)
X_test = agg[agg.columns.difference(['var16(t)'])].values
y_test = to_categorical(agg['var16(t)'].values)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

#X_train, y_train, X_test, y_test, X_validate, y_validate = tools.prepare_data_for_LSTM(train, test, validate)
lw = X_train.shape[1]
n_classes = max(train.status.nunique(), validate.status.nunique())

# create the model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(lw, input_shape=(X_train.shape[1], X_train.shape[2]), unroll=False, return_sequences=True))
model.add(keras.layers.LSTM(dense_width))
model.add(keras.layers.Dropout(dropout))
model.add(keras.layers.Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.categorical_accuracy])
print(model.summary())
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                    validation_data=(X_validate, y_validate),
                    verbose=1, shuffle=True,
                    callbacks=[early_stopping])
model.save(os.path.join(model_path, 'LSTM_model_%s_%d_%d_%.1f.h5'%(setnumber, dense_width, dense_depth, dropout)))
tools.plot_model(history, merge_data)
plt.show()

tools.make_prediction(model, X_test, y_test, fig=True)

## seq2seq
# train = pd.read_csv(os.path.join(data_path, 'train_'+setnumber+'_PC_merge.csv'), usecols=usecols)
# colname = 'RUL'
# if colname in train.columns:
#     train.drop(colname, 1, inplace=True)
# # Preprocess Data: (This does not work)
# wholeSequence = train.values
# data = wholeSequence[:-1] # all but last
# target = wholeSequence[1:] # all but first

# # Reshape training data for Keras LSTM model
# # The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# # Single batch, 9 time steps, 11 dimentions
# data = data.reshape((1, train.shape[0]-1, train.shape[1]))
# target = target.reshape((1, train.shape[0]-1, train.shape[1]))

# # Build Model
# model = Sequential()  
# model.add(LSTM(train.shape[1], input_shape=(train.shape[0]-1, train.shape[1]), unroll=True, return_sequences=True))
# model.add(Dense(train.shape[1]))
# model.compile(loss='mean_absolute_error', optimizer='adam')
# history = model.fit(data, target, epochs=200, batch_size=1,
#                     verbose=1, shuffle=True,
# )
# plt.plot(history.history['loss'])
# plt.show()
