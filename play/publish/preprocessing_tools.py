import sys
import os.path
from math import *
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.cross_decomposition
import sklearn.neighbors
import sklearn.model_selection
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
current_palette_4 = sns.color_palette("hls", 4)
sns.set_palette(current_palette_4)
import warnings
warnings.filterwarnings('ignore')
data_path = 'data'
plot_path = 'plots'

def read_set(data_path, setnumber, settype):
    name = os.path.join(data_path, settype+'_'+str(setnumber)+'.txt')
    data = pd.read_csv(name, delim_whitespace=True, header=None)
    new_cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s'+str(x) for x in range(1, 26-4)]
    data.columns = new_cols
    return data

#FIXME! relies on pd dtype identification (not reliable!) and currently not in use!
def find_col_types(df):
    id_columns = ['id', 'cycle']
    df_columns = df.columns.difference(id_columns)
    categorical_columns = [x for x in df_columns if df[x].dtype=='int']
    scalable_columns = [x for x in df_columns if x not in categorical_columns]
    return categorical_columns, scalable_columns

#sequences in training set stop by the full failure, i.e. RUL=0
def calculate_train_RUL(df):
    for part_id in df['id'].unique():
        max_cycle = df.loc[df['id']==part_id, 'cycle'].max()
        df.loc[df['id']==part_id,'RUL'] = max_cycle - df.loc[df['id']==part_id, 'cycle']
    return df

def calculate_test_RUL(df, label_df):
    for part_id in df['id'].unique():
        max_cycle = df.loc[df['id']==part_id, 'cycle'].max()
        label_RUL = label_df.loc[label_df['id']==part_id, 'RUL'].values[0]
        df.loc[df['id']==part_id,'RUL'] = max_cycle + label_RUL + (max_cycle - df.loc[df['id']==part_id, 'cycle'])
    return df

def plot_all_measurements(df, plot_path='plots', plot_name='raw_sequences.png'):
    cols = df.columns[2:26]
    fig, axs = plt.subplots(len(cols), figsize=(15, 10))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        axs[i].plot(df['cycle'], df[col], '-')
        h = axs[i].set_ylabel(col)
        h.set_rotation(0)
        axs[i].yaxis.set_label_position("right")
    plt.savefig(os.path.join(plot_path, plot_name))
    plt.show()
        
def plot_correlations(df, drop_cols=[], title='', plot_path='plots', plot_name='correlation.png'):
    tmp_df = df.drop(drop_cols, 1)
    corr = tmp_df.corr()
    plt.figure(figsize=(8, 8))
    g = sns.heatmap(corr)
    g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 8)
    g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 8)
    plt.title(title)
    plt.savefig(os.path.join(plot_path, plot_name))
    plt.show()



