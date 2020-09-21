import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


# 基本的线性模型
def linear_model(train, test):
    # 数据集划分
    x = train.drop(columns=['pm2.5_log']).values
    y = train['pm2.5_log'].values

    # pca降维处理
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca = pca.transform(x)
    x_tra, x_val, y_tra, y_val = train_test_split(x_pca, y, test_size=0.3, random_state=42)

    # 建立模型并验证mse
    reg = LinearRegression().fit(x_tra, y_tra)
    y_pre = reg.predict(x_val)
    print('Mean Squared Error: %.2f' % mean_squared_error(y_val, y_pre))

    # 预测
    X_test_pca = pca.transform(test)
    y_test = reg.predict(X_test_pca)
    y_real = np.round(np.exp(y_test))

    df = pd.DataFrame(y_real, columns=['pm2.5'])
    return df
