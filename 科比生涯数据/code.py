import time
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import KFold


'''
session_1           数据导入及可视化
session_2           特征提取
session_3           建模并分析
'''


def session_1():
    # 注意：pandas 通常不会完全显示
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('max_colwidth', 100)  # 设置 value 的显示长度为100，默认为50
    pd.set_option('display.width', 1000)  # 当 console 中输出的列数超过1000的时候才会换行

    # import data
    filename = "data/data.csv"
    raw = pd.read_csv(filename)
    print(raw.shape)
    print(raw.head())

    # 间数据分成两部分，一部分作为训练集，一部分为测试集
    # 测试数据集
    kobe = raw[pd.notnull(raw['shot_made_flag'])]
    print(kobe.shape)

    # 画图操作-投篮位置信息
    alpha = 0.02
    plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.scatter(kobe.loc_x, kobe.loc_y, color='R', alpha=alpha)
    plt.title('loc_x and loc_y')

    plt.subplot(122)
    plt.scatter(kobe.lon, kobe.lat, color='B', alpha=alpha)
    plt.title('lat and lon')

    plt.show()



def session_2():
    file_name = 'data/data.csv'
    raw = pd.read_csv(file_name)

    # 特征提取，将坐标转换为极坐标
    kobe = raw[pd.notnull(raw['shot_made_flag'])]
    raw['dist'] = np.sqrt(raw['loc_x'] ** 2 + raw['loc_y'] ** 2)

    loc_x_zero = raw['loc_x'] == 0

    raw['angle'] = np.array([0] * len(raw))
    # raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
    # raw['angle'][loc_x_zero] = np.pi / 2

    raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

    # unique 显示一列里所有不重复的值的集合
    print(kobe.action_type.unique())
    print(kobe.combined_shot_type.unique())

    print(kobe.shot_type.unique())
    print(kobe.shot_type.value_counts())

    print(kobe.season.unique())
    print(kobe.season.value_counts())

    # 画图
    plt.figure(figsize=(5, 5))

    plt.scatter(raw.dist, raw.shot_distance, color='blue')
    plt.title('dist and shot_distance')

    # 查看科比的投篮区域次数
    gs = kobe.groupby('shot_zone_area')
    print(kobe['shot_zone_area'].value_counts())
    print(len(gs))

    # 画图-对科比投篮的区域进行统计
    plt.figure(figsize=(20, 10))

    def scatter_plot_by_category(feat):
        alpha = 0.1
        gs = kobe.groupby(feat)
        cs = cm.rainbow(np.linspace(0, 1, len(gs)))
        for g, c in zip(gs, cs):
            plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)

    # shot_zone_area
    plt.subplot(131)
    scatter_plot_by_category('shot_zone_area')
    plt.title('shot_zone_area')

    # shot_zone_area
    plt.subplot(132)
    scatter_plot_by_category('shot_zone_basic')
    plt.title('shot_zone_basic')

    # shot_zone_range
    plt.subplot(133)
    scatter_plot_by_category('shot_zone_range')
    plt.title('shot_zone_range')

    plt.show()

    # 删除不重要的列，同时对一些字符进行one-hot处理
    drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'matchup',
             'lon', 'lat', 'seconds_remaining', 'minutes_remaining', 'shot_distance', 'loc_x', 'loc_y', 'game_event_id',
             'game_id', 'game_date']
    for drop in drops:
        raw = raw.drop(drop, 1)

    print(raw['combined_shot_type'].value_counts())

    # 制定前缀为 combined_shot_type, 查看前面两项数据
    x = pd.get_dummies(raw['combined_shot_type'], prefix='combined_shot_type')[0:2]
    print(x)



def session_3():
    file_name = 'data/data.csv'
    raw = pd.read_csv(file_name)

    # 删除不重要的列，同时对一些字符进行one-hot处理
    drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', 'matchup',
             'lon', 'lat', 'seconds_remaining', 'minutes_remaining', 'shot_distance', 'loc_x', 'loc_y', 'game_event_id',
             'game_id', 'game_date']
    for drop in drops:
        raw = raw.drop(drop, 1)

    # one-hot(独热编码)
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
    for var in categorical_vars:
        raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)  # concat 为拼接操作
        raw = raw.drop(var, 1)

    # 至此数据的整理已经完成，下面开始训练模型，目的是判断科比是否可以进球
    # 这里把'shot_made_flag'里的5000个有缺失值得数据当做测试集
    train_kobe = raw[pd.notnull(raw['shot_made_flag'])]
    train_label = train_kobe['shot_made_flag']
    train_kobe = train_kobe.drop('shot_made_flag', 1)
    test_kobe = raw[pd.isnull(raw['shot_made_flag'])]
    test_kobe = test_kobe.drop('shot_made_flag', 1)

    print(train_label)

    # 用随机森林训练模型，为了方便，森林的宽度和深度用了3个值（1,10,100）
    print('Finding best n_estimators for RandomForestClassifier...')
    min_score = 100000
    best_n = 0
    scores_n = []
    range_n = np.logspace(0,2,num=3).astype(int)                              # 建造一个从1~100的等比数列
    kf = KFold(n_splits=10, shuffle=True)

    for n in range_n:
        print('the number of trees : {0}'.format(n))
        t1 = time.time()

        rfc_score = 0.
        rfc = RandomForestRegressor(n_estimators=n)                           # 随机森林分类器建立一个模型
        for train_k, test_k in kf.split(train_kobe):
            rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])      # 一部分为数据，一部分为标签

            # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
            pred = rfc.predict(train_kobe.iloc[test_k])                       # 对模型进行预测
            rfc_score += log_loss(train_label.iloc[test_k], pred) / 10        # 对模型进行评估
        scores_n.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_n = n
        t2 = time.time()
        print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
    print(best_n, min_score)

    # find the best max_depth for RandomForestClassifier
    print('Finding best max_depth for RandomForestClassifier...')

    min_score = 100000
    best_m = 0
    scores_m = []
    range_m = np.logspace(0, 2, num=3).astype(int)
    kf = KFold(n_splits=10, shuffle=True)

    for m in range_m:
        print("the max depth : {0}".format(m))
        t1 = time.time()

        rfc_score = 0.
        rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
        for train_k, test_k in kf.split(train_kobe):
            rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
            # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10

            pred = rfc.predict(train_kobe.iloc[test_k])
            rfc_score += log_loss(train_label.iloc[test_k], pred) / 10

        scores_m.append(rfc_score)
        if rfc_score < min_score:
            min_score = rfc_score
            best_m = m

        t2 = time.time()
        print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))

    print(best_m, min_score)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(range_n, scores_n)
    plt.ylabel('score')
    plt.xlabel('number of trees')

    plt.subplot(122)
    plt.plot(range_m, scores_m)
    plt.ylabel('score')
    plt.xlabel('max depth')
    plt.show()

    model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
    model.fit(train_kobe, train_label)



if __name__ == '__main__':
    session_1()
    # session_2()
    # session_3()
