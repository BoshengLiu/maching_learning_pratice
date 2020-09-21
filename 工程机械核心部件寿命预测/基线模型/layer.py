from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class Layer:
    def __init__(self, n_estimators, num_forests, max_depth=100, min_samples_leaf=1):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = []


    def train(self, train_data, train_label, weight, val_data):
        val_prob = np.zeros([self.num_forests, val_data.shape[0]])
       
        for forest_index in range(self.num_forests):
            if forest_index % 2 == 0:
                clf = RandomForestRegressor(n_estimators=self.n_estimators,
                                             n_jobs=-1, 
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label, weight)
                val_prob[forest_index, :] = clf.predict(val_data)
            else:
                clf = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                           n_jobs=-1, 
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label, weight)
                val_prob[forest_index, :] = clf.predict(val_data)

            self.model.append(clf)

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0))
        return [val_avg, val_concatenate]


    def predict(self, test_data):
        predict_prob = np.zeros([self.num_forests, test_data.shape[0]])
        for forest_index, clf in enumerate(self.model):
            predict_prob[forest_index, :] = clf.predict(test_data)
        
        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0))
        return [predict_avg, predict_concatenate]


class KfoldWarpper:
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, max_depth=31, min_samples_leaf=1):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = []

    def train(self, train_data, train_label, weight):
        num_samples, num_features = train_data.shape

        val_prob = np.empty([num_samples])
        val_prob_concatenate = np.empty([num_samples, self.num_forests])

        # 这里要修改
        for train_index, test_index in self.kf.split(train_data):
            X_train = train_data[train_index, :]
            X_val = train_data[test_index, :]
            y_train = train_label[train_index]
            weight_train = weight[train_index]

            layer = Layer(self.n_estimators, self.num_forests, self.max_depth, self.min_samples_leaf)
            val_prob[test_index], val_prob_concatenate[test_index, :] = \
                layer.train(X_train, y_train, weight_train, X_val)
            self.model.append(layer)
        return [val_prob, val_prob_concatenate]


    def predict(self, test_data):
    
        test_prob = np.zeros([test_data.shape[0]])
        test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = \
                layer.predict(test_data)

            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold

        return [test_prob, test_prob_concatenate]
