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
import LSTM_tools as tools
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)

## Settings
merge_data = True
sn = 3
usecols = None
setnumber = 'FD00' + str(sn)
data_path = 'data'
plot_path = 'plots'
model_path = 'models'

## LSTM settings
n_in = 1
n_out = 1
dropout = 0.3
bs = 1 #batch size - the smaller the batch size, the longer it takes to finish an epoch of training
epochs = 200
LSTM_units = 9

## Alternative 1
train, test, validate = tools.read_preprocessed_data(setnumber, usecols, merge_data)
agg = tools.series_to_supervised(train, n_in=n_in, n_out=n_out)
X_train = agg[agg.columns.difference(['var16(t)'])].values
y_train = keras.utils.np_utils.to_categorical(agg['var16(t)'].values)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
agg = tools.series_to_supervised(validate, n_in=n_in, n_out=n_out)
X_validate = agg[agg.columns.difference(['var16(t)'])].values
y_validate = keras.utils.np_utils.to_categorical(agg['var16(t)'].values)
X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
agg = tools.series_to_supervised(test, n_in=n_in, n_out=n_out)
X_test = agg[agg.columns.difference(['var16(t)'])].values
y_test = keras.utils.np_utils.to_categorical(agg['var16(t)'].values)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
lw = X_train.shape[1]
n_classes = max(train.status.nunique(), validate.status.nunique())
# create the model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(
    lw,
    input_shape=(X_train.shape[1], X_train.shape[2]),
#    return_sequences=True
))
model.add(keras.layers.Dropout(dropout))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[keras.metrics.categorical_accuracy])
print(model.summary())
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                    validation_data=(X_validate, y_validate),
                    verbose=1, shuffle=True,
                    callbacks=[early_stopping])
model.save(os.path.join(model_path, 'LSTM_model_%s_%d_%.1f.h5'%(setnumber, LSTM_units, dropout)))
aucs, y_pred_bin = tools.make_prediction(model, X_test, y_test, fig=False)
y_pred = keras.utils.np_utils.to_categorical(np.array([np.argmax(y_pred_bin[i]) for i in range(len(y_pred_bin))]),
                                             num_classes=train.status.nunique())
print (sklearn.metrics.classification_report(y_test, y_pred))
