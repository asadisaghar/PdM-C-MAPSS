from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.gaussian_process as gp

def read_set(setnumber, settype):
    name = settype+'_'+str(setnumber)+'.txt'
    data = pd.read_csv(name, delim_whitespace=True, header=None)
    new_cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s'+str(x) for x in range(1, 26-4)]
    data.columns = new_cols
    return data

def find_col_types(df, id_columns):
    df_columns = df.columns.difference(id_columns)
    categorical_columns = [x for x in df_columns if df[x].dtype=='int']
    scalable_columns = [x for x in df_columns if x not in categorical_columns]
    return categorical_columns, scalable_columns

def scale(df, scalable_columns):
    scaler = StandardScaler()
    df[scalable_columns] = scaler.fit_transform(df[scalable_columns])
    return df

def onehot(df, categorical_columns):
    for col in categorical_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df

def calculate_RUL(df):
    for part_id in df['id'].unique():
        max_cycle = df.loc[df['id']==part_id, 'cycle'].max()
        df.loc[df['id']==part_id,'RUL'] = max_cycle - df.loc[df['id']==part_id, 'cycle']
    return df
    
def plot_all_measurements(df):
    cols = df.columns[2:26]
    fig, axs = plt.subplots(len(cols))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        if (col != 's17') and (col != 's18'):
            axs[i].plot(df['cycle'], df[col], label=col)
            axs[i].legend(loc=1)
    plt.show()

def plot_model(ax, x, y_pred, sigma, ob=''):
    """Plot the best-fit model of data along with 95% uncertainties."""
    
    ax.plot(x, y_pred, '-', c='k', linewidth=2, label='Prediction ' + ob)
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, color='darkorange', ec='None',
            label='95% confidence interval ' + ob)
    return ax

setnumber = 'FD004'
train = read_set(setnumber, 'train')
id_columns = ['id', 'cycle']
categorical_cols, scalable_cols = find_col_types(train, id_columns)
train = scale(train, scalable_cols)
#train = onehot(train, categorical_cols)
train = calculate_RUL(train)

## Try Support vector regressors
import sklearn.svm as svm

ten_train = train[train.id==10]
ten_train.drop('id', 1, inplace=True)
y = ten_train['RUL'].values.reshape((len(ten_train), 1))
ten_train.drop('RUL', 1, inplace=True)
X = ten_train

clf = svm.SVR(C=100, epsilon=0)
clf.fit(X, y)
print clf.score(X, y)

setnumber = 'FD004'
test = read_set(setnumber, 'test')
id_columns = ['id', 'cycle']
categorical_cols, scalable_cols = find_col_types(test, id_columns)
test = scale(test, scalable_cols)
#test = onehot(test, categorical_cols)

ten_test = test[test.id==10]
ten_test.drop('id', 1, inplace=True)
RUL = clf.predict(ten_test)

"""
## Try independent GPs
def make_gp_model(df, col):
    X = df['cycle'].values.reshape((354,1))
    y = df[col].values.reshape((354,1))

    gp1 = gp.GaussianProcess(
        theta0=df['cycle'].max()*10.,
        # thetaL=0.1,
        # thetaU=1,
        corr='squared_exponential',
        regr = "quadratic",
        random_start=500,
    )
    gp1.fit(X, y)
    y_pred, MSE = gp1.predict(X, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return X, y, y_pred, sigma

def plot_single_gps(df):
    cols = df.columns[2:26]
    fig, axs = plt.subplots(len(cols))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        if (col != 's17') and (col != 's18'):
            X, y, y_pred, sigma = make_gp_model(ten, col)
            axs[i].plot(X, y, 'o-')
            axs[i] = plot_model(axs[i], X, y_pred, sigma, '10-' + str(col))
#            axs[i].set_title('10-' + str(col))

    plt.show()

#plot_single_gps(train[train.id==10])

def make_coupled_gp_model(df, col1, col2):
    kernel = gp.kernels.ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(0.0, 10.0)) * gp.kernels.RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + gp.kernels.RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
    
    for hyperparameter in kernel.hyperparameters:
        print(hyperparameter)

    params = kernel.get_params()
    
    for key in sorted(params):
        print("%s : %s" % (key, params[key]))

"""
