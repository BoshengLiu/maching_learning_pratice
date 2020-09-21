from model_func import *
from data_process import *

if __name__ == '__main__':
    df_train = pd.read_csv('data/pm25_train.csv', low_memory=False)
    df_test = pd.read_csv('data/pm25_test.csv', low_memory=False)

    # 数据预处理
    df_train = dataProcess_train(df_train)
    print(df_train.columns)
    df_test = dataProcess_test(df_test)

    # 将数据类型改为float32
    for i in df_train.columns:
        df_train[i] = df_train[i].astype('float32')
    for j in df_test.columns:
        df_test[j] = df_test[j].astype('float32')

    # 模型选择
    df=linear_model(df_train, df_test)
    df.to_csv('sample.csv', encoding='utf-8', header=1, index=0)
