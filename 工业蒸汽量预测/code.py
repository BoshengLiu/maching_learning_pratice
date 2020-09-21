import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# 归一化函数
def dataFormat(df):
    for i in df.columns:
        df = df.copy()
        df[i] = (df[i].max() - df[i]) / (df[i].max() - df[i].min())
    return df


# 数据预处理，这里还可以增加一些特征，比如每小时采集频率得到的新特征
def dataProcess():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    # 线性相关系数绝对值小于0.2的列，可以选择删除，但是这些列可能存在非线性关系，尽量不删
    # columns = ['v19', 'V22', 'V26', 'V25', 'V21', 'V34', 'V14', 'V32', 'V33',
    #            'V28', 'V17', 'V29', 'V9', 'V35', 'V15', 'V18', 'V30', 'V13']

    label = pd.DataFrame(df_train['target'].values, columns=['target'])
    df_train.drop(columns=['target'], inplace=True)

    # 数据归一化
    train_new = dataFormat(df_train)
    test_new = dataFormat(df_test)

    dataView_factorLabel(train_new, label)
    labelRelation(train_new, label)

    label.to_csv('data/label.csv', index=False)
    train_new.to_csv('data/train_NN.csv', index=0)
    test_new.to_csv('data/test_NN.csv', index=0)


# 可视化，观察每个特征和target的关系，并保存为图片
def dataView_factorLabel(df, dt):
    for i in df.columns:
        plt.figure(figsize=(12, 8))
        plt.title(str(i) + '-target Distribution')
        plt.xlabel(str(i), rotation=45)
        plt.ylabel('target', rotation=45)
        plt.scatter(df[i].values, dt['target'].values)
        plt.savefig('factor_label/' + str(i) + '_target' + '.jpg')
        plt.clf()
        plt.close()  # 要加这个，消除警告


# 观察各个特征间的关系
def labelRelation(df, dt):
    # 查看正态分布及线性分布的可能性
    sns.distplot(dt.values, fit=norm)
    plt.show()
    plt.figure()
    probplot(dt['target'], plot=plt)
    plt.show()

    # 查看正负相关的20个特征
    df_new = pd.concat([df, dt], axis=1)
    corr = df_new.corr()
    k = 20
    cols_pos = corr.nlargest(k, 'target')['target'].index
    cols_neg = corr.nsmallest(k, 'target')['target'].index
    cols = cols_pos.append(cols_neg)

    cm = np.corrcoef(df_new[cols].values.T)
    sns.set(rc={'figure.figsize': (20, 16)})
    sns.set(font_scale=1.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig('person_coefficient.jpg')
    plt.show()


# 建立神经网络模型
def modelSetup():
    df_train = pd.read_csv('data/train_NN.csv')
    df_test = pd.read_csv('data/test_NN.csv')
    df_target = pd.read_csv('data/label.csv')

    for i in df_train.columns:
        df_train[i] = df_train[i].astype('float32')
    for i in df_test.columns:
        df_test[i] = df_test[i].astype('float32')

    # 建立神经网络模型
    model = Sequential()
    input_size = len(df_train.columns)
    
    # 参数是随便设置的，后续要进行调参，同时不确定是否有过拟合，Dropout先保留
    model.add(Dense(units=90, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.5))
    model.add(Dense(units=45, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(units=1, activation=None))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])

    y = df_target.values
    X = df_train.values

    # 划分训练集和测试集，选择25%为测试集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=40)

    # earlyStopping获取最佳模型，这个和Dropout一样防止过拟合，可以不用
    best_weights_file = 'best_weight.hdf5'
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    save_model = ModelCheckpoint(best_weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # 训练模型
    history = model.fit(X_train, y_train, batch_size=100, epochs=1000, verbose=1, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, save_model])

    # save model: early stopping
    model.load_weights(best_weights_file)

    # 可视化
    plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], label='train')
    plt.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'], label='valid')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0)
    plt.show()

    result = pd.DataFrame({'target': model.predict(df_test).reshape(1, -1)[0]})
    result.to_csv('sample.txt', index=False, header=None)

if __name__ == '__main__':
    # dataProcess()
    modelSetup()
