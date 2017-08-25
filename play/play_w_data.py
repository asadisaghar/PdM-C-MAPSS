from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

setnumber = 'FD004'
train = read_set(setnumber, 'train')
id_columns = ['id', 'cycle']
categorical_cols, scalable_cols = find_col_types(train, id_columns)
train = scale(train, scalable_cols)
train = onehot(train, categorical_cols)

