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
import keras.models
import keras.layers
import keras.optimizers
import keras.callbacks
import keras.utils.np_utils
import keras.preprocessing.sequence
from keras.objectives import categorical_crossentropy
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras import backend as K

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
dense_depth = 2
dropout = 0.5
bs = 128 #batch size
epochs = 5

## MLP tools
## Read preprocessed data
if merge_data:
    full = pd.read_csv(os.path.join(data_path, 'full_'+setnumber+'_PC_merge.csv'))
    train = pd.read_csv(os.path.join(data_path, 'train_'+setnumber+'_PC_merge.csv'), usecols=usecols)
    test = pd.read_csv(os.path.join(data_path, 'test_'+setnumber+'_PC_merge.csv'), usecols=usecols)
    validate = pd.read_csv(os.path.join(data_path, 'validate_'+setnumber+'_PC_merge.csv'), usecols=usecols)
else:
    full = pd.read_csv(os.path.join(data_path, 'full_'+setnumber+'_PC.csv'), usecols=usecols)    
    train = pd.read_csv(os.path.join(data_path, 'train_'+setnumber+'_PC.csv'), usecols=usecols)
    test = pd.read_csv(os.path.join(data_path, 'test_'+setnumber+'_PC.csv'), usecols=usecols)
    validate = pd.read_csv(os.path.join(data_path, 'validate_'+setnumber+'_PC.csv'), usecols=usecols)
colname = 'RUL'
if colname in train.columns:
    full.drop(colname, 1, inplace=True)
    train.drop(colname, 1, inplace=True)
    test.drop(colname, 1, inplace=True)
    validate.drop(colname, 1, inplace=True)

## Encode labels to integers
le = sklearn.preprocessing.LabelEncoder().fit(full.status)
full['status'] = le.transform(full.status)
train['status'] = le.transform(train.status)
test['status'] = le.transform(test.status)
validate['status'] = le.transform(validate.status)

## Split feature-label arrays
X_full = full[full.columns.difference(['id', 'cycle', 'status'])].values
y_full = keras.utils.np_utils.to_categorical(full['status'].values)

X_train = train[train.columns.difference(['id', 'cycle', 'status'])].values
y_train = keras.utils.np_utils.to_categorical(train['status'].values)

X_validate = validate[validate.columns.difference(['id', 'cycle', 'status'])].values
y_validate = keras.utils.np_utils.to_categorical(validate['status'].values)

X_test = test[test.columns.difference(['id', 'cycle', 'status'])].values
y_test = keras.utils.np_utils.to_categorical(test['status'].values)    

COLUMNS = full.columns
FEATURES = full.columns.difference(['id', 'cycle', 'status'])
LABEL = "status"

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

lw = len(FEATURES)
n_classes = train.status.nunique()

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, lw))
# Keras layers can be called on TensorFlow tensors:
x = keras.layers.Dense(dense_width, activation='relu')(img)  # fully-connected layer with "dense_width" units and ReLU activation
x = keras.layers.Dropout(dropout)(x)
layer = keras.layers.Dense(dense_width, activation='relu')
x = layer(x)
x = keras.layers.Dropout(dropout)(x)
preds = keras.layers.Dense(n_classes, activation='softmax')(x)  # output layer with "n_classes" units and a softmax activation

labels = tf.placeholder(tf.float32, shape=(None, n_classes))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)
# Initialize all variables
init_op = tf.global_variables_initializer()

# Run training loop
#sess = tf.Session('grpc://207.154.206.7:4711')
#with tf.Session('grpc://207.154.206.7:4711') as sess:
with tf.Session() as sess:
    K.set_session(sess)
    sess.run(init_op)
    for i in range(int(len(X_train)/bs)):
#        print i
        feature_batch = X_train[i*bs:(i+1)*bs]
        label_batch = y_train[i*bs:(i+1)*bs]
        train_step.run(
            feed_dict={
                img: feature_batch,
                labels: label_batch,
                K.learning_phase(): 1
            }
        )
#        print layer.trainable_weights  # list of TensorFlow Variables from an intermediate layer
    acc_value = keras.metrics.categorical_accuracy(labels, preds)
    print 'acc: ', np.sum(acc_value.eval(
        feed_dict={
            img: X_validate,
            labels: y_validate,
            K.learning_phase(): 0
        }
    ))/len(y_validate)
    
