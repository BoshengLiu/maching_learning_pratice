import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# 数据可视化
def pmShow(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df['PM2.5'].values, label='PM2.5')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('pm25.csv')

    pmShow(df)

    # 建立特征-天数索引和小时，生成日期，提取月份、日期
    df['day_index'] = [i // 24 + 1 for i in range(len(df))]
    df['hour'] = [i % 24 for i in range(len(df))]
    df['date'] = pd.date_range(start='2017-01-01', periods=len(df), freq='h')
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    del df['date']

    # 建立季节特征-春夏秋冬
    df['spring'] = [1 if (i == 3) | (i == 4) | (i == 5) else 0 for i in df['month'].values]
    df['summer'] = [1 if (i == 6) | (i == 7) | (i == 8) else 0 for i in df['month'].values]
    df['fall'] = [1 if (i == 9) | (i == 10) | (i == 11) else 0 for i in df['month'].values]
    df['winter'] = [1 if (i == 1) | (i == 2) | (i == 12) else 0 for i in df['month'].values]

    # 将缺失的序列作为测试集
    test = df[np.isnan(df['PM2.5'])]
    test = test.drop(["PM2.5"], axis=1)

    # 将缺失值删除，划分训练集合验证集
    df.dropna(axis=0, how='any', inplace=True)
    train = df.drop(["PM2.5"], axis=1)
    target = df['PM2.5']

    # 划分训练集和验证集
    x_tra, x_val, y_tra, y_val = train_test_split(train, target, test_size=0.3, random_state=42)

    # 建立模型
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(x_tra, y_tra)
    y_pred = rfr.predict(x_val)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print("RMSE is :%.6f." % rmse)

    # 预测
    y_test = rfr.predict(test)
    test['PM2.5'] = y_test

    # 连接训练集和测试集
    df_new = pd.merge(df, test, how='outer', on=df.columns.to_list(), sort='index')

    # 缺失值填充完可视化
    pmShow(df_new)

    test.to_csv('predict.csv', columns=['index', 'PM2.5'], index=False)
