import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot

"""---------------- 数据预处理 ----------------"""


# 时间数据数据转换
def timeFormat(dt):
    time_list = []
    t = time.strptime(dt, '%Y-%m-%d')
    time_list.append(t.tm_year)
    time_list.append(t.tm_mon)
    time_list.append(t.tm_mday)
    time_list.append(t.tm_wday)
    return time_list


# 可视化操作
def dataPlot(dt):
    for i in dt.columns:
        plt.figure(figsize=(16, 5))
        plt.title(str(i) + '-' + 'pm2.5')
        plt.plot(dt[i].values, dt['pm2.5_log'].values)
        plt.show()


# 线性分布检验和正态分布检验
def dataAnalyse(dt):
    # pm2.5正态分布可能性
    sns.distplot(dt['pm2.5'], fit=norm)
    plt.title("Normal distribution")
    plt.show()
    print("Skewness: %f" % dt['pm2.5'].skew())
    print("Kurtosis: %f" % dt['pm2.5'].kurt())

    # pm2.5线性分布可能性
    plt.figure()
    plt.title('Linear possibility')
    probplot(dt['pm2.5'], plot=plt)
    plt.show()

    # 将pm2.5所在列进行处理，删除值为0的行
    df = dt.drop(dt[dt['pm2.5'] == 0].index)
    df['pm2.5_log'] = np.log(df['pm2.5'])

    # 观察经过log转换后pm2.5正态分布可能性
    sns.distplot(df['pm2.5_log'], fit=norm)
    plt.title("Normal distribution")
    plt.show()
    print("Skewness: %f" % df['pm2.5_log'].skew())
    print("Kurtosis: %f" % df['pm2.5_log'].kurt())

    # 观察经过log转换后pm2.5线性分布可能性
    plt.figure()
    plt.title('Linear possibility')
    probplot(df['pm2.5_log'], plot=plt)
    plt.show()


# 特征相关性检验
def dataCorrelation(df):
    # 查看正负相关的8个特征
    corrmat = df.corr()
    k = 8
    cols_pos = corrmat.nlargest(k, 'pm2.5_log')['pm2.5_log'].index
    cols_neg = corrmat.nsmallest(k, 'pm2.5_log')['pm2.5_log'].index
    cols = cols_pos.append(cols_neg)

    cm = np.corrcoef(df[cols].values.T)
    sns.set(rc={'figure.figsize': (12, 10)})
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


"""---------------- 处理训练集和测试集 ----------------"""


# 处理训练集
def dataProcess_train(dt):
    # 将时间数据连接到dataframe
    data = dt['date'].tolist()
    df_list = []
    for i in data:
        df_list.append(timeFormat(i))
    df_time = pd.DataFrame(df_list)
    df_time.columns = ['year', 'month', 'day', 'week']

    # 将两个dataframe连接起来
    df = pd.concat([df_time, dt], axis=1)
    df.drop(columns=['date'], inplace=True)

    # 数据分析及可视化
    dataAnalyse(df)

    # 删除pm2.5值为0的行，并对其进行log处理
    df = df.drop(df[df['pm2.5'] == 0].index)
    df['pm2.5_log'] = np.log(df['pm2.5'])
    df.drop(columns=['pm2.5'], inplace=True)

    # 相关性检验
    dataCorrelation(df)

    return df


# 处理测试集
def dataProcess_test(dt):
    # 将时间数据转换
    data = dt['date'].tolist()
    df_list = []
    for i in data:
        df_list.append(timeFormat(i))
    df_time = pd.DataFrame(df_list)
    df_time.columns = ['year', 'month', 'day', 'week']

    # 将两个dataframe连接起来
    df = pd.concat([df_time, dt], axis=1)
    df.drop(columns=['date'], inplace=True)

    return df
