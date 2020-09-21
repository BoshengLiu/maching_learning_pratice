from __future__ import absolute_import, division, print_function
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import GridSpec
import seaborn as sns
import os, sys
from tqdm import tqdm
import warnings

import pandas_profiling
import missingno
from sklearn.datasets import make_blobs
import time

warnings.filterwarnings('ignore')
sns.set_context('poster', font_scale=1.3)

file_name = 'CrowdstormingData.csv'
df = pd.read_csv(file_name)

# print(df.head(10))
# print(df.describe().T)
# print(df.isnull().sum())
# print(df.dtypes)

# 查看列信息
columns = df.columns.tolist()
# print(columns)

# 统计身高均值
# print(df['height'].mean())
# print(np.mean(df.groupby('playerShort').height.mean())) # 统计均值，不重复
# print(np.mean(df.groupby('playerShort').weight.mean()))

# group by的例子
# df2 = pd.DataFrame({'key1':['a','b','a','a','b'],
#                     'key2':['one','two','two','two','one'],
#                     'data1':np.random.randn(5),
#                     'data2':np.random.randn(5)})
#
# print(df2)
# grouped = df2['data1'].groupby(df2['key1'])
# print(grouped.mean())

# Create Tidy Player Table
# 建立一个球员的信息表
player_index = 'playerShort'
player_cols = ['club', 'leagueCountry', 'birthday', 'height', 'weight',
               'photoID', 'rater1', 'rater2']

# 检测球员的错误统计数据
all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})
# print(all_cols_unique_players.head().T)
# print(all_cols_unique_players[all_cols_unique_players > 1].dropna().head())
# print(all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0]==0)

# 判断重复样本
def get_subgroup(dataframe, g_index, g_columns):
    g = dataframe.groupby(g_index).agg({col:'nunique' for col in g_columns})
    if g[g>1].dropna().shape[0] != 0:
        print("Warning: you probably assumed this had all unique values but it doesn't")
    return dataframe.groupby(g_index).agg({col:'max' for col in g_columns})

# 将数据进行处理
player = get_subgroup(df, player_index, player_cols)
# print(player.head())

# 将球员的信息进行保存
def save_subgroup(dataframe, g_index, subgroup_name, prefix='raw_'):
    save_subgroup_filename = ''.join([prefix, subgroup_name, '.csv.gz'])
    dataframe.to_csv(save_subgroup_filename, compression='gzip', encoding='utf-8')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip',
                          index_col=g_index, encoding='utf-8')

    if dataframe.equals(test_df):
        print('Test-passed: we recover the equivalent subgroup dataframe.')
    else:
        print('Warning -- equivalence test!!! Double-check.')
# save_subgroup(player, player_index, 'player')

# 俱乐部信息
club_index = 'club'
club_cols = ['leagueCountry']
clubs = get_subgroup(df, club_index, club_cols)
# print(clubs.head())
# print('-'*30)
# print(clubs['leagueCountry'].value_counts())

# 保存俱乐部信息
# save_subgroup(clubs, club_index, 'clubs')











