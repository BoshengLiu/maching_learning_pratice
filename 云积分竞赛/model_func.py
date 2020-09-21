import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from data_process import *
import warnings

warnings.filterwarnings('ignore')


def rfcTraining(x, y):
    def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
        val = cross_val_score(
            RandomForestClassifier(n_estimators=int(n_estimators),
                                   min_samples_split=int(min_samples_split),
                                   max_features=min(max_features, 0.999),
                                   max_depth=int(max_depth),
                                   random_state=2),
            x, y, scoring='roc_auc', cv=5).mean()
        return val

    rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (50,500),
         'min_samples_split': (10,300),
         'max_features': (8,15),
         'max_depth': (8, 15)}
    )
    rf_bo.maximize()
    print(rf_bo.max)



def xgbPredict(x, y, test):
    x_tra, x_val, y_tra, y_val = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    params = {'learning_rate': 0.1,
              'max_depth': 6,
              'num_boost_round': 5000,
              'objective': 'binary:logistic',
              'random_state': 7,
              'silent': 0,
              'eta': 0.8,
              'eval_metric': 'auc'
              }

    train = xgb.DMatrix(x_tra, y_tra)
    valid = xgb.DMatrix(x_val, y_val)
    x_test = xgb.DMatrix(test)
    eval = [(train, 'train'), (valid, 'valid')]

    best = xgb.train(params, train, evals=eval, early_stopping_rounds=50)
    y_test = best.predict(x_test, ntree_limit=best.best_ntree_limit)

    plot_importance(best, max_num_features=12)
    plt.show()

    return y_test
