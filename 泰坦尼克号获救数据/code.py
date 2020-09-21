import re
import numpy as np

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)

'''
session_1           数据可视化
session_2           数预处理
session_3           利用线性回归预测
session_4           利用逻辑回归预测
session_5           利用随机森林预测
session_6           自定义特征，利用随机森林预测
'''


def session_1():
    file_name = 'titanic_data.csv'
    df = pd.read_csv(file_name)
    cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']

    sns.barplot(x='Embarked', y='Survived', hue='sex', data=df)
    plt.title('不同性别在不同登舱口的生存情况')
    plt.show()

    sns.barplot(x='Pclass', y='Survived', hue='sex', data=df)
    plt.title('不同性别在不同等级的船舱的生存情况')
    plt.show()

    sns.barplot(x='Title', y='Survived', hue='sex', data=df)
    plt.title('不同性别在不同登舱口的生存情况')
    plt.show()

    sns.barplot(x='FamilySize', y='Survived', hue='sex', data=df)
    plt.title('不同性别在不同家庭人数的生存情况')
    plt.show()


def session_2():
    file_name = 'titanic_train.csv'
    df = pd.read_csv(file_name)

    # 数据预处理
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['NameLength'] = df['Name'].apply(lambda x: len(x))

    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ''

    titles = df['Name'].apply(get_title)
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9,
                     'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17}

    for k, v in title_mapping.items():
        titles[titles == k] = v

    df['Title'] = titles
    df['PassengerId'] = df['PassengerId']
    df['Survived'] = df['Survived']
    df['Pclass'] = df['Pclass']
    df['Fare'] = df['Fare']

    cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']
    df.to_csv('titanic_data.csv', index=False, columns=cols)


def session_3():
    df = pd.read_csv('titanic_train.csv')

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    # 设置标签
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    alg = LinearRegression()

    kf = KFold(n_splits=3, random_state=1)
    predictions = []
    for train, test in kf.split(df[predictors]):
        train_predictors = df[predictors].iloc[train, :]
        train_target = df['Survived'].iloc[train]

        alg.fit(train_predictors, train_target)

        # 对测试集进行预测
        test_predictions = alg.predict(df[predictors].iloc[test, :])

        predictions.append(test_predictions)

    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > 0.5] = 1  # 大于0.5表示获救
    predictions[predictions <= 0.5] = 0  # 小于0.5表示未获救

    accuracy = len(predictions[predictions == df['Survived']]) / len(predictions)

    print(accuracy)


def session_4():
    df = pd.read_csv('titanic_train.csv')

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # 版本问题，需要在后面添加 solver='liblinear'，否则会有警告，虽然不影响运行
    alg_1 = LogisticRegression(random_state=1, solver='liblinear')
    scores = model_selection.cross_val_score(alg_1, df[predictors], df['Survived'], cv=3)
    print(scores.mean())


def session_5():
    df = pd.read_csv('titanic_train.csv')

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    # 设置标签
    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # 通过随机森林进行预测，这里随机森林的参数只是随便设定的，具体参数需要建立一个随机森林模型
    alg_2 = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

    kf = model_selection.KFold(n_splits=3, random_state=1, shuffle=True)
    scores_1 = model_selection.cross_val_score(alg_2, df[predictors], df['Survived'], cv=kf)

    print(scores_1.mean())


def session_6():
    df = pd.read_csv('titanic_train.csv')

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    # 自己构建特征
    df['FamilySize'] = df['SibSp'] + df['Parch']  # 兄弟姐妹和老人小孩
    df['NameLength'] = df['Name'].apply(lambda x: len(x))  # 名字的长度

    # 每个人都有自己的身份的词，如Miss, Mr...
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ''

    titles = df['Name'].apply(get_title)

    # 将称号用数字表示
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 9,
                     'Countess': 10, 'Ms': 11, 'Lady': 12, 'Jonkheer': 13, 'Don': 14, 'Mme': 15, 'Capt': 16, 'Sir': 17}
    for k, v in title_mapping.items():
        titles[titles == k] = v

    df['Title'] = titles
    predictors_new = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']

    selector = SelectKBest(f_classif, k=5)
    selector.fit(df[predictors_new], df['Survived'])
    scores_2 = -np.log10(selector.pvalues_)

    plt.bar(range(len(predictors_new)), scores_2)
    plt.xticks(range(len(predictors_new)), predictors_new, rotation='vertical')
    plt.show()

    # 通过图像选出权重比高的5个特征
    predictors_1 = ['Pclass', 'Sex', 'Fare', 'Title', 'NameLength']
    alg_3 = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=2)

    kf = model_selection.KFold(n_splits=3, random_state=1, shuffle=True)
    scores_1 = model_selection.cross_val_score(alg_3, df[predictors_1], df['Survived'], cv=kf)

    print(scores_1.mean())


if __name__ == '__main__':
    session_1()
    session_2()
    session_3()
    session_4()
    session_5()
    session_6()
