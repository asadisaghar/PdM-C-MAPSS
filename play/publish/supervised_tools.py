import sys
import os.path
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.metrics
import keras.metrics
import keras.models
import keras.layers
import keras.optimizers
import keras.callbacks
import keras.utils.np_utils
import keras.preprocessing.sequence
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
data_path = 'data'
plot_path = 'plots'
model_path = 'models'

# KNN tools
def make_knn(n, X_train, y_train, X_test, y_test):
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    return model

def make_prediction_knn(model, X_test, y_test, fig=False):
    y_pred = model.predict(X_test)
    aucs = np.array([])
    if fig:
        plt.figure()
    for i in range(y_pred.shape[1]):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test[:,i], y_pred[:,i])
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs = np.append(aucs, auc)
        if fig:
            plt.plot(fpr, tpr, label='RUL| %d (auc: %.2f) '%(i+1, auc))
    if fig:
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()
    return aucs, y_pred

## MLP tools
def make_mlp(train, X_train, dense_width, dense_depth, dropout):
    lw = X_train.shape[1]
    n_classes = train.status.nunique()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(dense_width, activation='relu', input_dim=lw))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())
    for i in range(dense_depth):
        model.add(keras.layers.Dense(dense_width, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.BatchNormalization())
    if n_classes == 2:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=[keras.metrics.categorical_accuracy])
    return model

def make_hourglass(train, X_train, dropout=0.0):
    dense_width = 10
    lw = X_train.shape[1]
    n_classes = train.status.nunique()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(dense_width, activation='relu', input_dim=lw))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(int(dense_width/2), activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(int(dense_width/2), activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.BatchNormalization())
    if n_classes == 2:
        model.add(keras.layers.Dense(n_classes, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(keras.layers.Dense(n_classes, activation='softmax'))
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=[keras.metrics.categorical_accuracy])
    return model

def fit_mlp(model, X_train, y_train, X_validate, y_validate, bs, epochs):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, 
                        validation_data=(X_validate, y_validate), 
                        verbose=0, shuffle=True,
                        callbacks=[early_stopping])
    return history

def plot_learning_curve(history, setnumber, merge_data):
    fig, axs = plt.subplots(1,2, figsize=(15, 5))
    axs = axs.flatten()
    axs[0].plot(history.history['categorical_accuracy'], label='train')
    axs[0].plot(history.history['val_categorical_accuracy'], label='validation')
    axs[0].set_title('categorical_accuracy')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='validation')
    axs[1].set_title('loss')
    axs[1].legend()
    axs[1].grid()
    #axs[0].set_ylim(0.9, 1.0)
    #axs[1].set_ylim(0.0, 0.3)
    if merge_data:
        plt.savefig(os.path.join(plot_path, 'MLP_history_%s_merge.png'%setnumber))
    else:
        plt.savefig(os.path.join(plot_path, 'MLP_history_%s_set.png'%setnumber))
    plt.show()

## Shared tools
def make_prediction(model, X_test, y_test, fig=False):
    y_pred = model.predict(X_test)
    aucs = np.array([])
    if fig:
        plt.figure()
    for i in range(y_pred.shape[1]):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test[:,i], y_pred[:,i])
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs = np.append(aucs, auc)
        if fig:
            plt.plot(fpr, tpr, label='RUL %d (auc: %.2f) '%(i+1, auc))
    if fig:
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.grid()
        plt.show()
    return aucs, y_pred

def prepare_data(setnumber, usecols, merge_data):
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
    if 'RUL' in train.columns:
        train.drop('RUL', 1, inplace=True)
        test.drop('RUL', 1, inplace=True)
        validate.drop('RUL', 1, inplace=True)    
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
    
    print X_train.shape, X_validate.shape, X_test.shape
    return X_train, y_train, X_test, y_test, X_validate, y_validate, train, test, validate
