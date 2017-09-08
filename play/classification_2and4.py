import sys
import numpy
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import  sklearn.metrics
from keras import regularizers

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load dataset
#sn = 3
#epochs = 500
sn = int(sys.argv[1])
group = int(sys.argv[2])
epochs = int(sys.argv[3])
dense_depth = 1
        
setnumber = 'FD00' + str(sn)
if group==0:
        dataframe = pandas.read_csv('data/PCA/train_'+setnumber+'.csv')
else:
        dataframe = pandas.read_csv('data/PCA/train_'+setnumber+'_group'+str(group)+'.csv')
dataframe.dropna(inplace=True)
dataset = dataframe.values
if sn==2:
        X = dataset[:,5:-2]
        Y = dataset[:,-1]
elif sn==4:
        X = dataset[:,4:-2].astype(float)
        Y = dataset[:,-1]

test_dataframe = pandas.read_csv('data/PCA/test_'+setnumber+'.csv')
test_dataframe.dropna(inplace=True)
test_dataset = test_dataframe.values
if sn==2:
        X_test = test_dataset[:,5:-2]
        Y_test = test_dataset[:,-1]
elif sn==4:
        X_test = test_dataset[:,4:-2].astype(float)
        Y_test = test_dataset[:,-1]

# [one-hot] encode class values
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

encoded_Y_test = encoder.transform(Y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

# define baseline model
def baseline_model(dense_depth, sn, X):
	# create model
	model = Sequential()
        if sn == 2:
	        # model.add(Dense(2, input_dim=17,
                #                 activation='relu',
                #                 use_bias=True,
                #                 bias_initializer='zeros',
                #                 bias_regularizer=regularizers.l2(0.5),
                #                 # kernel_initializer='random_uniform',
                #                 # kernel_regularizer=regularizers.l2(0.5),
                # ))
                model.add(Dense(1, input_dim=X.shape[1], kernel_initializer='normal', use_bias=True, activation='relu'))                
                model.add(Dropout(0.5))
                model.add(Dense(1, kernel_initializer='normal', use_bias=True, activation='relu'))
                model.add(Dropout(0.5))
        elif sn == 4:
	        model.add(Dense(dense_depth, input_dim=18,
                                activation='relu',
                                use_bias=True,
                                bias_initializer='zeros',
                                bias_regularizer=regularizers.l2(0.05),
                                kernel_initializer='random_uniform',
                                kernel_regularizer=regularizers.l1(0.01),
                ))
                model.add(Dropout(0.2))
                
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model(dense_depth, sn, X)
history = model.fit(X, dummy_y, epochs=epochs, batch_size=200, validation_data=(X_test, dummy_y_test), verbose=1, shuffle=True)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#plt.show()
plt.savefig('plots/PCA/history_Dense_%d_%s_%depochs.png'%(dense_depth, setnumber, epochs))

# make a prediction
yhat = model.predict(X_test)
plt.figure()
for i in range(yhat.shape[1]):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(dummy_y_test[:,i], yhat[:,i])
        auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='class %d (auc: %.2f) '%(i+1, auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
#plt.show()
plt.savefig('plots/PCA/ROC_Dense_%d_%s_%depochs.png'%(dense_depth, setnumber, epochs))

# #Use SKlearn cross-validation
# estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=50, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print results
