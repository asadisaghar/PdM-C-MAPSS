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
    encoder = sklearn.preprocessing.LabelEncoder()
    for col in cat_columns:
        values[:,col] = encoder.fit_transform(values[:,col])
    # endure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
    return scaler

# set the dataset
sn = str(sys.argv[1])
# set batch size
bs = int(sys.argv[2])
#set number of epochs
epoch = int(sys.argv[3])
#set first layer width
lw = int(sys.argv[4])
#set number of stacked LSTM layers
stack_depth = int(sys.argv[5])

setnumber = 'FD00' + str(sn)
scaler = more_preprocessing('train', setnumber, [21, 22])
train_scaled = scaler.fit_transform(values)
# frame as supervised learning
train = series_to_supervised(train_scaled, n_in=1, n_out=1, dropnan=True).values

test, test_scaler = more_preprocessing('test', setnumber, [21, 22])
test_scaled = scaler.fit_transform(values)
# frame as supervised learning
test = series_to_supervised(test_scaled, n_in=1, n_out=1, dropnan=True).values

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = keras.models.Sequential()
for i in range(stack_depth-1):
        model.add(keras.layers.LSTM(lw, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(keras.layers.LSTM(lw, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(keras.layers.Dense(1, activation='softplus')) #more stable compared to 'relu'...
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epoch, batch_size=bs, validation_data=(test_X, test_y), verbose=1, shuffle=True)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat))
#rmse = sqrt(sklearn.metrics.mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)

plt.title('Test RMSE: %.3f' % rmse)
plt.savefig('plots/'+setnumber+'_BatchSize'+str(bs)+'_Epochs'+str(epoch)+'_LayerWidth'+str(lw)+'_Stack'+str(stack_depth)+'.png')
