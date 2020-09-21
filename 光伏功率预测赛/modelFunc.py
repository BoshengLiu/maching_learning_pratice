import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from hyperopt import fmin, tpe, hp, partial
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from dataPreprocess import *
import warnings

warnings.filterwarnings('ignore')


# lightGBM自动调参
def lgbTraining(x_train, y_train, p):
    train_x, valid_x, train_y, valid_y = train_test_split(x_train.values, y_train.values, test_size=0.3,
                                                          random_state=42)
    train = lgb.Dataset(train_x, train_y)
    valid = lgb.Dataset(valid_x, valid_y, reference=train)

    # 自定义hyperopt的参数空间
    space = {"max_depth": hp.randint("max_depth", 15),
             "num_trees": hp.randint("num_trees", 20),
             'learning_rate': hp.randint('learning_rate', 20),
             "num_leaves": hp.randint("num_leaves", 10),
             "lambda_l1": hp.randint("lambda_l1", 6)
             }

    def argsDict_tranform(argsDict, isPrint=False):
        argsDict["max_depth"] = argsDict["max_depth"] + 10
        argsDict["num_trees"] = argsDict["num_trees"] * 5 + 100
        argsDict["learning_rate"] = argsDict["learning_rate"] * 0.01 + 0.01
        argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
        argsDict["lambda_l1"] = argsDict["lambda_l1"] * 0.1
        if isPrint:
            print(argsDict)
        else:
            pass
        return argsDict

    def lightgbm_factory(argsDict):
        argsDict = argsDict_tranform(argsDict)
        params = {'nthread': -1,  # 进程数
                  'max_depth': argsDict['max_depth'],  # 最大深度
                  'num_trees': argsDict['num_trees'],  # 树的数量
                  'learning_rate': argsDict['learning_rate'],  # 学习率
                  'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
                  'lambda_l1': argsDict["lambda_l1"],  # L1 正则化
                  'lambda_l2': 0,  # L2 正则化
                  'objective': 'regression',
                  'bagging_seed': 100  # 随机种子,light中默认为100
                  }
        params['metric'] = ['mae']
        model_lgb = lgb.train(params, train, num_boost_round=20000, valid_sets=[valid], early_stopping_rounds=100)
        return get_transformer_score(model_lgb)

    # 获取实际功率大于0.03*p的部分
    valid_y_new = valid_y[valid_y > 0.03 * p]
    valid_y_new_index = np.argwhere(valid_y > 0.03 * p)

    def get_transformer_score(transformer):
        model = transformer
        prediction = model.predict(valid_x, num_iteration=model.best_iteration)
        prediction_new = prediction[valid_y_new_index]
        return mean_absolute_error(valid_y_new, prediction_new)

    # 开始使用hyperopt进行自动调参
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(lightgbm_factory, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None)
    MAE = lightgbm_factory(best) / p

    return MAE, best


def lgbPredict(x_train, y_train, x_test, params):
    train_x, valid_x, train_y, valid_y = train_test_split(x_train.values, y_train.values, test_size=0.3,
                                                          random_state=42)
    train = lgb.Dataset(train_x, train_y)
    valid = lgb.Dataset(valid_x, valid_y, reference=train)

    fit_params = {
        'num_boost_round': 20000,
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }
    params['metric'] = ['mae']
    params['objective'] = ['regression']

    lgb_reg = lgb.train(params, train, valid_sets=[valid], **fit_params)
    y_pred = lgb_reg.predict(x_test, num_iteration=lgb_reg.best_iteration)
    return y_pred


def stackModel(x_train, y_train, x_test):
    x_all = pd.concat([x_train, x_test], ignore_index=True)
    # 简单归一化处理
    x_std_tools = MinMaxScaler(feature_range=(-1, 1))
    x_std_tools.fit(x_all)
    x_all = x_std_tools.transform(x_all)
    x_train_std = x_std_tools.transform(x_train)
    x_test_std = x_std_tools.transform(x_test)

    # PCA降维处理
    pca = PCA(n_components=12)
    pca.fit(x_all)
    x_train_pca = pca.transform(x_train_std)
    x_test_pca = pca.transform(x_test_std)

    # 模型融合
    m1 = BaggingRegressor(LinearRegression(), n_estimators=100, n_jobs=3)
    m2 = AdaBoostRegressor(LinearRegression(), n_estimators=100)
    m3 = BaggingRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100, n_jobs=3)
    m4 = AdaBoostRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100)
    m5 = BaggingRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100, n_jobs=3)
    m6 = AdaBoostRegressor(DecisionTreeRegressor(min_samples_split=500), n_estimators=100)
    m7 = BaggingRegressor(ExtraTreeRegressor(min_samples_split=500), n_estimators=100, n_jobs=3)
    m8 = AdaBoostRegressor(ExtraTreeRegressor(min_samples_split=500), n_estimators=10)
    m9 = RandomForestRegressor(n_estimators=100, min_samples_split=500, n_jobs=3)
    m10 = GradientBoostingRegressor(n_estimators=100, min_samples_split=500)
    models = VotingRegressor([('m1', m1), ('m2', m2), ('m3', m3), ('m4', m4), ('m5', m5),
                              ('m6', m6), ('m7', m7), ('m8', m8), ('m9', m9), ('m10', m10)], n_jobs=3)
    y_predict = []
    models.fit(x_train_pca, y_train)
    test_predict = models.predict(x_test_pca)
    y_predict.append(test_predict)

    return y_predict


lgb_params = [
    {'lambda_l1': 0.3, 'learning_rate': 0.18, 'max_depth': 10, 'num_leaves': 37, 'num_trees': 195},
    {'lambda_l1': 0.4, 'learning_rate': 0.2, 'max_depth': 9, 'num_leaves': 34, 'num_trees': 170},
    {'lambda_l1': 0.0, 'learning_rate': 0.15, 'max_depth': 10, 'num_leaves': 37, 'num_trees': 195},
    {'lambda_l1': 0.4, 'learning_rate': 0.18, 'max_depth': 7, 'num_leaves': 34, 'num_trees': 195},
    {'lambda_l1': 0.2, 'learning_rate': 0.2, 'max_depth': 8, 'num_leaves': 37, 'num_trees': 150},
    {'lambda_l1': 0.4, 'learning_rate': 0.19, 'max_depth': 7, 'num_leaves': 37, 'num_trees': 190},
    {'lambda_l1': 0.4, 'learning_rate': 0.17, 'max_depth': 10, 'num_leaves': 37, 'num_trees': 195},
    {'lambda_l1': 0.3, 'learning_rate': 0.2, 'max_depth': 10, 'num_leaves': 28, 'num_trees': 195},
    {'lambda_l1': 0.0, 'learning_rate': 0.2, 'max_depth': 8, 'num_leaves': 37, 'num_trees': 185},
    {'lambda_l1': 0.5, 'learning_rate': 0.17, 'max_depth': 9, 'num_leaves': 37, 'num_trees': 190}
]
