import sys
import os.path
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.gaussian_process as gp
import sklearn.metrics
import keras.models
import keras.layers
import keras.preprocessing.sequence
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)

data_path = 'data'
plot_path = 'plots'
model_path = 'models'

## Tools
def prepare_data_for_LSTM(train, test, validate, n_in, n_out):
    maxlen = int(max(train.cycle.max(), test.cycle.max()))
    print('maxlen: %d'%maxlen)
    if n_in == 0:
        n_in = maxlen
        
    # frame as supervised learning
    train_framed = pd.DataFrame()
    for i in train.id.unique():
        t_in = pad_columns(train.loc[train.id==i], maxlen)
        t_in.drop(['id', 'cycle'], 1, inplace=True)
        tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
        train_framed = train_framed.append(tmp)

    drop_cols = ['var%d(t)'%(i) for i in range(1,train.shape[1]-2)]
    train_framed.drop(drop_cols, 1, inplace=True)

    test_framed = pd.DataFrame()
    for i in test.id.unique():
        t_in = pad_columns(test.loc[test.id==i], maxlen)
        t_in.drop(['id', 'cycle'], 1, inplace=True)
        tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
        test_framed = test_framed.append(tmp)
        
    test_framed.drop(drop_cols, 1, inplace=True)

    validate_framed = pd.DataFrame()
    for i in validate.id.unique():
        t_in = pad_columns(validate.loc[validate.id==i], maxlen)
        t_in.drop(['id', 'cycle'], 1, inplace=True)
        tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
        validate_framed = validate_framed.append(tmp)

    drop_cols = ['var%d(t)'%(i) for i in range(1,validate.shape[1]-2)]
    validate_framed.drop(drop_cols, 1, inplace=True)
    X_train = train_framed.values[:,:-1]
    y_train = train_framed.values[:,-1]
    X_test = test_framed.values[:,:-1]
    y_test = test_framed.values[:,-1]
    X_validate = validate_framed.values[:,:-1]
    y_validate = validate_framed.values[:,-1]
    # pad input sequences
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, dtype='float64', maxlen=maxlen)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, dtype='float64', maxlen=maxlen)
    X_validate = keras.preprocessing.sequence.pad_sequences(X_validate, dtype='float64', maxlen=maxlen)
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
    return X_train, y_train, X_test, y_test, X_validate, y_validate

##The old function!
# def prepare_data_for_LSTM(train, test, validate, n_in, n_out):
#     maxlen = int(max(train.cycle.max(), test.cycle.max()))
#     # frame as supervised learning
#     train_framed = pd.DataFrame()
#     for i in train.id.unique():
#         t_in = train.loc[train.id==i, train.columns.difference(['id', 'cycle'])]
#         tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
#         train_framed = train_framed.append(tmp)

#     drop_cols = ['var%d(t)'%(i) for i in range(1,train.shape[1]-2)]
#     train_framed.drop(drop_cols, 1, inplace=True)

#     test_framed = pd.DataFrame()
#     for i in test.id.unique():
#         t_in = test.loc[test.id==i, test.columns.difference(['id', 'cycle'])]
#         tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
#         test_framed = test_framed.append(tmp)
#     test_framed.drop(drop_cols, 1, inplace=True)

#     validate_framed = pd.DataFrame()
#     for i in validate.id.unique():
#         t_in = validate.loc[train.id==i, validate.columns.difference(['id', 'cycle'])]
#         tmp = series_to_supervised(t_in.values, n_in, n_out, dropnan=True)
#         validate_framed = validate_framed.append(tmp)

#     drop_cols = ['var%d(t)'%(i) for i in range(1,validate.shape[1]-2)]
#     validate_framed.drop(drop_cols, 1, inplace=True)
#     X_train = train_framed.values[:,:-1]
#     y_train = train_framed.values[:,-1]
#     X_test = test_framed.values[:,:-1]
#     y_test = test_framed.values[:,-1]
#     X_validate = validate_framed.values[:,:-1]
#     y_validate = validate_framed.values[:,-1]
#     print('pre padding:\n')
#     print(X_train.shape, X_test.shape, X_validate.shape)
#     # pad input sequences
#     X_train = keras.preprocessing.sequence.pad_sequences(X_train, dtype='float64', maxlen=maxlen)
#     X_test = keras.preprocessing.sequence.pad_sequences(X_test, dtype='float64', maxlen=maxlen)
#     X_validate = keras.preprocessing.sequence.pad_sequences(X_validate, dtype='float64', maxlen=maxlen)
#     print('\npost padding:\n')
#     print(X_train.shape, X_test.shape, X_validate.shape)
#     # reshape input to be 3D [samples, timesteps, features]
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
#     X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
#     X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
#     return X_train, y_train, X_test, y_test, X_validate, y_validate

# convert series to supervised learning
def read_preprocessed_data(setnumber, usecols, merge_data):
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

    return train, test, validate

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    No identifier column (id or cycle) should be sent to this function!
    given dropnan=True this is how rows and columns of the output(o) of this fnuction compare to those of the input(i)
    row_o = row_i - n_in
    col_o = (col_i) * (n_in + n_out)
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# pad columns - use for each sample sequence separately
def pad_columns(df, maxlen, mode='pre'):
        if maxlen < df.cycle.max():
                print 'the "maxlen" you chose is shorter than the current sequence. Use truncate_columns instead to truncate the sequence.'
                return df
        else:
                if mode == 'pre':
                        tmp = pd.DataFrame({'cycle': range(int(df.cycle.max()), maxlen), 'id': df.id.unique()[0]}, columns=df.columns) #FIXME! possible off-by-one error in using range!
                        tmp = df.append(tmp).fillna(-1)
                elif mode == 'post':
                        pass #FIXME!
                return tmp

# pad columns - use for each sample sequence separately
def truncate_columns(df, minlen, mode='post'):
        # if minlen > df.cycle.max():
        #         print 'the "minlen" you chose is longer than the current sequence. Use pad_column instead to zero-pad the sequence.', df.id.unique()
        #         return df
        # else:
                if mode == 'post':
                        tmp = df.loc[df.cycle<=minlen]
                elif mode == 'pre':
                        pass
                return tmp
def plot_learning_curve(history, merge_data):
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
        plt.savefig(os.path.join(plot_path, 'MLP_history_merge.png'))
    else:
        plt.savefig(os.path.join(plot_path, 'MLP_history_set.png'))
        plt.show()
        
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
    else:
        return None, y_pred
