import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
# 注：会出现警告，我们可以使用上面的代码来忽视它


'''
session_1           特征提取及标准化处理
session_2           过采样处理
session_3           数据处理及建模
'''


def session_1():
    # 注意：pandas 通常不会完全显示
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)     # 显示所有行
    pd.set_option('max_colwidth', 100)          # 设置 value 的显示长度为100，默认为50
    pd.set_option('display.width', 1000)        # 当 console 中输出的列数超过1000的时候才会换行

    # import data
    data = pd.read_csv('creditcard.csv')
    print(data.head())

    count_classes = pd.value_counts(data['Class'], sort=True).sort_index()  # 查看该列有多少种不同的属性值
    count_classes.plot(kind='bar')

    # 作图
    plt.title('Fraud class histogram')  # 标题
    plt.xlabel('Class')                 # x轴添加名称
    plt.xticks(rotation=45)             # 将x轴数据旋转90°
    plt.ylabel('Frequency')             # y轴添加名称

    plt.show()

    # 标准化处理
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    # fit_transform 对数据进行变换
    # 注：这里 reshape 前面要加'.value'

    data = data.drop(['Time', 'Amount'], axis=1)  # 去除不需要的特征
    print(data.head())



def session_2():
    # 数据读取及划分
    credit_card = pd.read_csv('data/creditcard.csv')

    # 建模，Recall = TP/(TP+FN)
    def printing_Kfold_scores(x_train_data, y_train_data):
        fold = KFold(5, shuffle=True)

        c_param_range = [0.01, 0.1, 1, 10, 100]  # 正则化惩罚向（惩罚力度）

        results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
        results_table['C_parameter'] = c_param_range

        j = 0
        # 每次循环使用不同的惩罚参数，选出最优的一个
        for c_param in c_param_range:
            print('-' * 30)
            print('C parameter: ', c_param)
            print('-' * 30)
            print('')

            recall_accs = []
            # 交叉验证
            for iteration, indices in enumerate(fold.split(x_train_data)):
                # 使用C参数调用回归模型
                lr = LogisticRegression(C=c_param, penalty='l1')

                # 使用训练数据来拟合模型
                lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

                # 使用训练数据中的测试指数来进行预测
                y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

                # 计算召回率并添加到列表中
                recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)

                recall_accs.append(recall_acc)
                print('Iteration', iteration, ': recall score = ', recall_acc)

            # 召回分数的值是我们想要保存和掌握的指标
            results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
            j += 1
            print('')
            print('Mean recall score ', np.mean(recall_accs))
            print('')

        best_c = results_table.loc[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']
        # 注：idxmax()前要加‘.astype('float64')’

        print('*' * 30)
        print('Best model to choose from cross validation is with C parameter =', best_c)
        print('*' * 30)
        return best_c

    # 混合矩阵
    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    columns = credit_card.columns
    feathres_columns = columns.delete(len(columns) - 1)

    feathres = credit_card[feathres_columns]
    labels = credit_card['Class']

    feathres_train, feathres_test, labels_train, labels_test = train_test_split(
        feathres, labels, test_size=0.2, random_state=0
    )

    # SMOTE 处理训练集
    oversample = SMOTE(random_state=0)
    os_feathres, os_labels = oversample.fit_resample(feathres_train, labels_train)
    print(len(os_labels[os_labels == 1]))

    # 交叉验证
    os_feathres = pd.DataFrame(os_feathres)
    os_labels = pd.DataFrame(os_labels)
    best_c = printing_Kfold_scores(os_feathres, os_labels)

    # 混合矩阵
    lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
    lr.fit(os_feathres, os_labels.values.ravel())
    y_pred = lr.predict(feathres_test.values)

    cnf_maxtrix = confusion_matrix(labels_test, y_pred)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset:", cnf_maxtrix[1, 1] / (cnf_maxtrix[1, 0] + cnf_maxtrix[1, 1]))

    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_maxtrix, classes=class_names, title='Confusion matrix')
    plt.show()



def session_3():
    data = pd.read_csv('data/creditcard.csv')
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    # 下采样处理，使得0（正样品）和1（负样品）数据一样少
    # 注：ix 已经被弃用，可以使用 loc 或者 iloc
    X = data.loc[:, data.columns != 'Class']
    y = data.loc[:, data.columns == 'Class']

    # 计算出负样品的样本数，并获取它们的索引，转换成 array 格式
    number_records_fraud = len(data[data.Class == 1])
    fraud_indices = np.array(data[data.Class == 1].index)

    # 获取正样品的索引
    normal_indices = data[data.Class == 0].index

    # 在正样品的索引索引中随机选择样本，样本数为 number_records_fraud，然后获取新的索引，转换成 array 格式
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    # 将两个样本合并在一起
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    # 经过下采样所拿到的数据集
    under_sample_data = data.iloc[under_sample_indices]

    # 下采样数据集的数据
    X_under_samples = under_sample_data.loc[:, under_sample_data.columns != 'Class']
    y_under_samples = under_sample_data.loc[:, under_sample_data.columns == 'Class']

    # 正样品数
    print("Percentage of normal transactions: ",
          len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))

    # # 负样品数
    print("Percentage of fraud transactions: ",
          len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))

    # # 总样品数
    print("Total number of transactions in resampled data: ", len(under_sample_data))

    # 交叉验证，将数据切分成测试集和训练集，测试数据集设为0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("Number transactions train dataset: ", len(X_train))
    print("Number transactions train dataset: ", len(y_test))
    print("Total number of transactions: ", len(X_train) + len(X_test))

    # 下采样数据集
    X_train_undersample, X_test_unsersample, y_train_undersample, y_test_undersample = train_test_split(X_under_samples,
                                                                                                        y_under_samples,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)

    print('')
    print("Number transcations train dataset: ", len(X_train_undersample))
    print("Number transcations test dataset: ", len(X_test_unsersample))
    print("Total number of transactions: ", len(X_train_undersample) + len(X_test_unsersample))


    # 建模，Recall = TP/(TP+FN)
    def printing_Kfold_scores(x_train_data, y_train_data):
        fold = KFold(5, shuffle=True)

        c_param_range = [0.01, 0.1, 1, 10, 100]  # 正则化惩罚向（惩罚力度）

        results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
        results_table['C_parameter'] = c_param_range

        j = 0
        # 每次循环使用不同的惩罚参数，选出最优的一个
        for c_param in c_param_range:
            print('-' * 30)
            print('C parameter: ', c_param)
            print('-' * 30)
            print('')

            recall_accs = []
            # 交叉验证
            for iteration, indices in enumerate(fold.split(x_train_data)):
                # 使用C参数调用回归模型
                lr = LogisticRegression(C=c_param, penalty='l1')

                # 使用训练数据来拟合模型
                lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

                # 使用训练数据中的测试指数来进行预测
                y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

                # 计算召回率并添加到列表中
                recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)

                recall_accs.append(recall_acc)
                # print('Iteration', iteration,': recall score = ',recall_acc)

            # 召回分数的值是我们想要保存和掌握的指标
            results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
            j += 1
            print('')
            print('Mean recall score ', np.mean(recall_accs))
            print('')

        best_c = results_table.loc[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']
        # 注：idxmax()前要加‘.astype('float64')’

        print('*' * 30)
        print('Best model to choose from cross validation is with C parameter =', best_c)
        print('*' * 30)
        return best_c

    best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)
    print(best_c)

    # 混合矩阵
    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    lr = LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_train, y_train.values.ravel())

    y_pred = lr.predict(X_test.values)
    # lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    # y_pred_undersamples = lr.predict(X_test_unsersample.values)

    # 计算全数据集混合矩阵
    cnf_matrix = confusion_matrix(y_test, y_pred)

    # 计算低采样数据集混合矩阵
    # cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersamples)

    np.set_printoptions(precision=2)

    print('Recall metric in the testing dataset: ', cnf_matrix)

    # 非归一化混合矩阵
    class_name = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_name, title='Confusion matrix')
    plt.show()

    lr = LogisticRegression(C=0.01, penalty='l1')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample_proba = lr.predict_proba(X_test_unsersample.values)  # 这里改成计算结果的概率值

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.figure(figsize=(10, 10))

    # 将预测的概率值与阈值进行对比
    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
        np.set_printoptions(precision=2)

        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold > %s' % i)
    plt.show()


if __name__ == '__main__':
    # session_1()
    # session_2()
    session_3()
