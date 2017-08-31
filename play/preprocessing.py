import sys
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_set(setnumber, settype):
    name = 'original_data/'+settype+'_'+str(setnumber)+'.txt'
    data = pd.read_csv(name, delim_whitespace=True, header=None)
    new_cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s'+str(x) for x in range(1, 26-4)]
    data.columns = new_cols
    return data

def find_col_types(df, id_columns):
    df_columns = df.columns.difference(id_columns)
    categorical_columns = [x for x in df_columns if df[x].dtype=='int']
    scalable_columns = [x for x in df_columns if x not in categorical_columns]
    return categorical_columns, scalable_columns

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

def plot_correlations(df, drop_cols=[], title='', plot_name='correlation'):
    tmp_df = df.drop(drop_cols, 1)
    corr = tmp_df.corr()
    plt.figure()
    g = sns.heatmap(corr)
    g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 8)
    g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 8)
    plt.title(title)
    plt.savefig('plots/'+plot_name+'.png')
    
sn = str(sys.argv[1])
setnumber = 'FD00'+sn
id_columns = ['id', 'cycle']

#Training set
train = read_set(setnumber, 'train')
cat_train, scale_train = find_col_types(train, id_columns)
train = calculate_train_RUL(train)
train.to_csv('data/train_'+setnumber+'.csv')

#Test set
test = read_set(setnumber, 'test')
cat_test, scale_test = find_col_types(test, id_columns)

#Labels
label = pd.read_csv('original_data/'+'RUL_'+setnumber+'.txt', header=None)
label.reset_index(level=[0], inplace=True)
label.columns = ['id', 'RUL']
label['id'] = label['id'] + 1  #index is 0-bound but part_ids are 1-bound

test = calculate_test_RUL(test, label)
test.to_csv('data/test_'+setnumber+'.csv')

plt.figure()
sns.distplot(train.RUL, label='train')
sns.distplot(test.RUL, label='test')
plt.legend()
plt.title('RUL distribution for various engines (ids)')
#plt.show()
plt.savefig('plots/'+setnumber+'_RULs.png')

plot_correlations(train, drop_cols=['id', 'cycle', 'RUL'], title=setnumber, plot_name=setnumber)
