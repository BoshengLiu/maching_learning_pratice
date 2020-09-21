import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')


# 获取数据地址
def getFile(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFile(newDir, fileList)

    return fileList


# 统计数据
def countData(data, count, name):
    count[name + '_max'] = data.max()
    count[name + '_min'] = data.min()
    count[name + '_count'] = data.count()
    count[name + '_mean'] = data.mean()
    count[name + '_ptp'] = data.ptp()
    count[name + '_std'] = data.std()
    return count


# 处理单个样本
def process_single_sample(data, p):
    df = pd.read_csv(data)
    lifeMax = df['部件工作时长'].values.max()
    df = df[df['部件工作时长'] <= lifeMax * p]
    count = {
        'train_file_name': os.path.basename(data) + str(p),
        '开关1_sum': df['开关1信号'].values.sum(),
        '开关2_sum': df['开关2信号'].values.sum(),
        '告警1_sum': df['告警信号1'].values.sum(),
        '设备': df['设备类型'][0],
        'life': lifeMax - df['部件工作时长'].values.max()}
    columns = ['部件工作时长', '累积量参数1', '累积量参数2', '转速信号1', '转速信号2',
               '压力信号1', '压力信号2', '温度信号', '流量信号', '电流信号']
    for i in columns:
        count = countData(df[i], count, i)
    features = pd.DataFrame(count, index=[0])

    return features


idx = 'train_file_name'
y_col = 'life'


# 多进程调用单文件处理函数，并整合到一起
def multiProgress(cpu, fileList, isTest, func):
    if isTest:
        train_p = [1]
        rst = []
        pool = Pool(cpu)

        for file in fileList:
            for i in train_p:
                rst.append(pool.apply_async(func, args=(file, i,)))
        pool.close()
        pool.join()

        rst = [i.get() for i in rst]
        features = rst[0]
        for i in rst[1:]:
            features = pd.concat([features, i], axis=0)
        cols = features.columns.tolist()
        for col in [idx, y_col]:
            cols.remove(col)
        cols = [idx] + cols + [y_col]

        features[idx] = features[idx].apply(lambda x: x[:-1])
        features = features.reindex(columns=cols)
    else:
        train_p = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        rst = []
        pool = Pool(cpu)

        for file in fileList:
            for j in train_p:
                rst.append(pool.apply_async(func, args=(file, j,)))
        pool.close()
        pool.join()

        rst = [i.get() for i in rst]
        features = rst[0]
        for i in rst[1:]:
            features = pd.concat([features, i], axis=0)
        cols = features.columns.tolist()
        for col in [idx, y_col]:
            cols.remove(col)
        cols = [idx] + cols + [y_col]
        features = features.reindex(columns=cols)

    return features


# 设置评价指标
def lossCalculate(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res


# lgb
def lgbCV(train, params, fit_params, feature_names, n_fold, seed, test):
    train_pre = pd.DataFrame({'true': train[y_col], 'pred': np.zeros(len(train))})
    test_pre = pd.DataFrame({idx: test[idx], y_col: np.zeros(len(test))}, columns=[idx, y_col])
    k_folder = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

    for fold_id, (tra_idx, val_idx) in enumerate(k_folder.split(train)):
        print(f'\nFold_{fold_id} Training...\n')

        lgb_tra = lgb.Dataset(
            data=train.iloc[tra_idx][feature_names],
            label=train.iloc[tra_idx][y_col],
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx][y_col],
            feature_name=feature_names)

        lgb_reg = lgb.train(params=params, train_set=lgb_tra, **fit_params, valid_sets=[lgb_tra, lgb_val])
        val_pre = lgb_reg.predict(train.iloc[val_idx][feature_names], num_iteration=lgb_reg.best_iteration)
        train_pre.loc[val_idx, 'pred'] = val_pre
        test_pre[y_col] += lgb_reg.predict(test[feature_names]) / n_fold

    score = lossCalculate(train_pre['true'], train_pre['pred'])
    print('\nCV LOSS:', score)
    return test_pre


# lgb参数设置
params_lgb = {
    'num_leaves': 500,
    'max_depth': 10,
    'learning_rate': 0.02,
    'objective': 'regression',
    'boosting': 'gbdt',
    'verbosity': -1}

fit_params_lgb = {
    'num_boost_round': 5000,
    'verbose_eval': 200,
    'early_stopping_rounds': 200}

# 主程序
if __name__ == '__main__':
    start = time.time()

    # 获取数据地址
    train_list = getFile('train/', [])
    test_list = getFile('test1/', [])

    n = 4
    func = process_single_sample
    train = multiProgress(n, train_list, False, func)
    test = multiProgress(n, test_list, True, func)
    print("process done: " + str(time.time() - start) + " sec.")

    train_test = pd.concat([train, test], join='outer', axis=0).reset_index(drop=True)
    train_test = pd.get_dummies(train_test, columns=['设备'])
    feature_name = list(filter(lambda x: x not in [idx, y_col], train_test.columns))

    sub = lgbCV(train_test.iloc[:train.shape[0]], params_lgb, fit_params_lgb, feature_name, 5, 2018,
                train_test.iloc[train.shape[0]:])
    sub.to_csv('sample.csv', index=False)
    print("process done: " + str(time.time() - start) + " sec.")
