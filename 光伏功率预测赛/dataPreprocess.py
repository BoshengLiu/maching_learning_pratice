import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from luminol.anomaly_detector import AnomalyDetector
from lightgbm import LGBMRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False

'''     数据预处理     '''


# 时间格式处理
def date_format(date):
    time_list = []
    dt_new = pd.to_datetime(date)
    t = time.strptime(str(dt_new), '%Y-%m-%d %H:%M:%S')
    time_list.append(t.tm_year)
    time_list.append(t.tm_mon)
    time_list.append(t.tm_mday)
    time_list.append(t.tm_hour)
    time_list.append(t.tm_min)
    return time_list


# 地方时角，正午时角为0，6点为-90°，18点为90°
def divide_hourly_angle(dt_hour, dt_min):
    length = len(dt_hour)
    dt_hour = dt_hour.values
    dt_min = dt_min.values
    pi = np.pi

    hour_angle = np.zeros((length,))
    for i in range(length):
        hour_angle[i] = dt_hour[i] * pi / 12 - pi

    min_angle = np.zeros((length,))
    for i in range(length):
        if dt_min[i] == 15:
            min_angle[i] = pi / 48
        elif dt_min[i] == 30:
            min_angle[i] = pi / 36
        elif dt_min[i] == 45:
            min_angle[i] = pi / 24
        else:
            min_angle[i] = 0

    time_angle = hour_angle + min_angle
    return time_angle


# 积日公式
def calculate_accumulate(year, month, day):
    y = int(year)
    m = int(month)
    d = int(day)
    if y % 4 == 0 and y % 100 != 0 or y % 400 == 0:
        leap = 1
    else:
        leap = 0
    Accumulate = (0, 31, leap + 59, leap + 90, leap + 120, leap + 151, leap + 181,
                  leap + 212, leap + 243, leap + 273, leap + 304, leap + 334)
    result = Accumulate[m - 1] + d
    return result


# 赤纬算法
def calculate_declination(year, month, day):
    length = len(year)
    year = year.values
    month = month.values
    day = day.values
    ED = np.zeros((length,))  # 赤纬角度
    for i in range(length):
        N = calculate_accumulate(year[i], month[i], day[i])
        N0 = 79.6764 + 0.2422 * (year[i] - 1985) - int((year[i] - 1985) / 4)
        theta = 2 * (np.pi) * (N - N0) / 365.2422
        delta = 0.3723 + 23.2567 * np.sin(theta) + 0.1149 * np.sin(2 * theta) - 0.1712 * np.sin(3 * theta) - \
                0.758 * np.cos(theta) + 0.3656 * np.cos(2 * theta) + 0.0201 * np.cos(3 * theta)
        ED[i] = abs(delta)
    direct_latitude = ED * np.pi / 180
    return direct_latitude


# 计算太阳高度角
def calculate_sunAngle(dt_time_angle, dt_direct_latitude):
    length = len(dt_time_angle)
    direct_latitude = dt_direct_latitude.values
    elevate_angle = dt_time_angle.values
    high_angle = np.zeros((length,))
    phi = np.pi / 10
    for i in range(length):
        high_angle[i] = np.sin(phi) * np.sin(direct_latitude[i]) + np.cos(phi) \
                        * np.cos(direct_latitude[i]) * np.cos(elevate_angle[i])
    return high_angle


# 数据预处理
def processData(df):
    # 将时间提取出来，转化为列表形式
    sample_date = df['时间'].values.tolist()
    # 初始化一个空列表
    date_list = []
    # 遍历date列，并将其格式转换为年、月、日、时、分
    for i in sample_date:
        date_list.append(date_format(i))  # 向列表添加年、月、日、时、分
    # 将列表转换为DataFrame
    sample_time = pd.DataFrame(date_list)
    # 给DataFrame添加列名
    sample_time.columns = ['年', '月', '日', '时', '分']
    # 重新建立索引
    df = df.reset_index(drop=True)
    # 连接两个DataFrame
    df_new = pd.concat([sample_time, df], axis=1)
    # 将时间数据类型转换为int32
    for i in sample_time.columns:
        df_new[i] = df_new[i].astype('int32')

    # 获取时角
    df_new['时角'] = divide_hourly_angle(df_new['时'], df_new['分'])
    # 获取赤纬角度
    df_new['直射纬度'] = calculate_declination(df_new['年'], df_new['月'], df_new['日'])
    # 获取太阳高度角
    df_new['高度角'] = calculate_sunAngle(df_new['时角'], df_new['直射纬度'])

    df_new.drop(columns=['年', '月', '日', '时', '分', '直射纬度'], inplace=True)
    df_new = df_new.reset_index(drop=True)
    return df_new


# 数据处理
def cleanData(df, index_name="15分钟段", var_name="实际功率", limit=0.5):
    df_clean = []
    for g_name, g in df.groupby(index_name):
        temp = deepcopy(g).reset_index(drop=True)
        limit_low, limit_up = np.percentile(temp[var_name], [5, 95])
        temp = temp[(temp[var_name] < limit_up) & (temp[var_name] > limit_low)].reset_index(drop=True)
        ts = temp[var_name]
        ts_mean = np.mean(ts)
        ts_std = np.std(ts)
        ts = (ts - ts_mean) / ts_std
        if ts_std > 0:
            my_detector = AnomalyDetector(ts.to_dict(), algorithm_name='exp_avg_detector')
            score = my_detector.get_all_scores()
            df_clean.append(temp[np.array(score.values) < limit])
        else:
            df_clean.append(temp)
    df_clean = pd.concat(df_clean, ignore_index=True)
    return df_clean


'''     数据可视化操作     '''


# 清洗前数据可视化
def saveBeforeclean(df, num):
    plt.figure(figsize=[72, 8])
    sns.boxplot(x="15分钟段", y="实际功率", data=df)
    plt.savefig("plot/boxplot_%02d.png" % num)
    plt.close('all')


# 清洗后数据可视化
def saveAfterclean(df, num):
    plt.figure(figsize=[72, 8])
    sns.boxplot(x="15分钟段", y="实际功率", data=df)
    plt.savefig("plot/boxplot_clear_%02d.png" % num)
    plt.close('all')


# 预测结果可视化
def predictShow(predict, num):
    plt.figure()
    pd.Series(predict).plot()
    plt.savefig("plot/predict_%02d.png" % num)
    plt.close('all')


# 特征选择
def featureChoice(train, test, num):
    feature = train
    label = test

    model = LGBMRegressor(n_estimators=100, num_leaves=30, max_depth=8)
    model.fit(feature.values, label.values)

    df_features = pd.DataFrame({'column': feature.columns,
                                'importance': model.feature_importances_}).sort_values(by='importance')
    # 可视化，并将结果保存到本地
    plt.figure(figsize=(24, 16))
    plt.barh(range(len(df_features)), df_features['importance'], height=0.8, alpha=0.6)
    plt.yticks(range(len(df_features)), df_features['column'])
    plt.title("第" + str(num) + "个电场的 feature importance")
    plt.savefig('feature/' + '电场_' + str(num) + '.jpg')

    feature_choice = df_features['column'].values.tolist()
    return feature_choice[-24:]  # 选择特征分数前24的特征
