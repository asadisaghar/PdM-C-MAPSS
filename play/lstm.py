import sys
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

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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


def more_preprocessing(settype, setnumber, cat_columns=[]):
    df = pd.read_csv('data/'+settype+'_'+setnumber+'.csv')
    values = df.values

    # integer encode categorical columns
    # FIXME! OneHotEncode these too...
    # encoder = sklearn.preprocessing.LabelEncoder()
    # for col in cat_columns:
    #     values[:,col] = encoder.fit_transform(values[:,col])
    
    # endure all data is float
    values = values.astype('float64')

    # normalize features
    if settype == 'train':
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(values)
    else:
        scaler = None

    # pad sequences
    sequences = [df.loc[df.id==i] for i in df.id.unique()]
    padded = keras.preprocessing.sequence.pad_sequences(sequences, padding='post', dtype='float64')
    df_padded = pd.DataFrame(columns=df.columns)
    for i in range(len(padded)):
        df_padded = df_padded.append(pd.DataFrame(data=padded[i], columns=df.columns))
    df = df_padded
    
    return df, scaler

def pad_columns(df, maxlen):
        tmp = pd.DataFrame({'cycle': range(df.cycle.max(), maxlen), 'id': df.id.unique()[0]}, columns=df.columns) #FIXME! possible off-by-one error in using range!
        tmp = df.append(tmp).fillna(0)
        return tmp

# # set the dataset
# sn = str(sys.argv[1])
# # set the sample/engine ID
# sid = str(sys.argv[2])
# # set batch size
# bs = int(sys.argv[3])
# #set number of epochs
# epoch = int(sys.argv[4])
# #set first layer width
# lw = int(sys.argv[5])
# #set number of stacked LSTM layers
# stack_depth = int(sys.argv[6])

sn = 1
bs = 100
epoch = 5
lw = 2
stack_depth = 2

n_in = 10
n_out = 1

setnumber = 'FD00' + str(sn)

# read data
train = pd.read_csv('data/train_'+setnumber+'.csv')
test = pd.read_csv('data/test_'+setnumber+'.csv')

# drop the extra column
train.drop('Unnamed: 0', 1, inplace=True)
test.drop('Unnamed: 0', 1, inplace=True)

# normalize features
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train.values)
train_values = scaler.transform(train.values)
test_values = scaler.transform(test.values)

# pad sequences
maxlen = train.cycle.max()

padded = pd.DataFrame(columns=train.columns)
for i in train.id.unique():
    padded = padded.append(pad_columns(train.loc[train.id==i], maxlen))
train = padded
#train.drop(['id'], 1, inplace=True)

padded = pd.DataFrame(columns=train.columns)
for i in test.id.unique():
    padded = padded.append(pad_columns(test.loc[test.id==i], maxlen))
test = padded
#test.drop(['id'], 1, inplace=True)

# read the data dimensions for the LSTM layer
train_samples = train.id.nunique()
test_samples = test.id.nunique()

train_timesteps = int(train.cycle.max())
test_timesteps = int(test.cycle.max())

# frame as supervised learning
#FIXME! this needs to be done for each ID separately...
train = series_to_supervised(train, n_in, n_out, dropnan=True)
# drop_cols = range(train.shape[1]*n_in, train.shape[1]*(n_in+1)-1)
# train.drop(train.columns[drop_cols], axis=1, inplace=True)
test = series_to_supervised(test, n_in, n_out, dropnan=True)
# test.drop(test.columns[drop_cols], axis=1, inplace=True)

# train = train.values
# test = test.values

# split into input and outputs
train_X, train_y = train.values[:, :-1], train.values[:, -1]
test_X, test_y = test.values[:, :-1], test.values[:, -1]

train_features = train_X.shape[1]
test_features = test_X.shape[1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_samples, train_timesteps, train_features))
test_X = test_X.reshape((test_samples, test_timesteps, test_features))

# design network
model = keras.models.Sequential()

# normal LSTM layers where the internal state is updated only at the end of an epoch
for i in range(stack_depth-1):
        model.add(keras.layers.LSTM(lw, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(keras.layers.LSTM(lw, input_shape=(train_X.shape[1], train_X.shape[2])))

#FIXME! width of the final fully-connected output layer...
model.add(keras.layers.Dense(1, activation='softplus')) #more stable compared to 'relu'...
model.compile(loss='mae', optimizer='adam')
print model.summary()

# fit network
#history = model.fit(train_X, train_y, epochs=epoch, batch_size=bs, validation_data=(test_X, test_y), verbose=1, shuffle=True)
history = model.fit(train_X, train_y, epochs=epoch, batch_size=bs, verbose=1, shuffle=True)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# # # make a prediction
# # yhat = model.predict(test_X)

# # # calculate RMSE
# # rmse = sqrt(sklearn.metrics.mean_squared_error(test_y, yhat))
# # print('Test RMSE: %.3f' % rmse)

# # plt.title('Test RMSE: %.3f' % rmse)
# # plt.savefig('plots/'+setnumber+'_BatchSize'+str(bs)+'_Epochs'+str(epoch)+'_LayerWidth'+str(lw)+'_Stack'+str(stack_depth)+'.png')

# # plt.figure()
# # plt.plot(test_y, yhat, '.')
# # plt.plot(test_y, test_y, '-')
# # rmse = sqrt(sklearn.metrics.mean_squared_error(test_y, yhat))
# # plt.title('Test RMSE: %.3f' % rmse)
# # plt.savefig('plots/prediction.png')
