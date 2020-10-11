import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from Recommendation_recall import recommendation_base_on_MF, user_item_matrix


# 对于系统的排序阶段，我们通常是这样的：
# 以召回阶段的输出作为输入
# 用CTR预估作为进一步的排序标准
# 这里，我们可以召回50首音乐，用 GBDT+LR 对这些音乐做 ctr 预估，给出评分排序，选出10首歌曲。
# 现在，仅仅用用户-物品评分是不够的，因为我们需要考虑特征之间的组合。为此，我们用之前的 data 数据。
# 这里的数据处理思路是:
# 复制一份新的数据，命名为 new_data
# 去掉 title 列，因为它不需要参与特征组合
# 对其余 object 列进行 labelencoder 编码
# 根据 rating 列数值情况，为了样本的正负均衡，我们令 rating 小于0.7的为0，也就是不喜欢，令rating大于0.7的为1，也就是喜欢
# 将 new_data 按照0.5的比例分成两份，一份给 GBDT 作为训练集，一份给lr作为训练集

def data_process(data):
    df = data[['song', 'play_count']].groupby('song').sum()
    df.rename(columns={'play_count': 'all_counts'}, inplace=True)
    new_data = data.merge(df, left_on='song', right_on='song')
    new_data['rating'] = np.log(new_data['play_count'].values / new_data['all_counts'] + 2)
    new_data.drop('title', axis=1, inplace=True)
    new_data['rating'] = new_data['rating'].apply(lambda x: 0 if x < 0.7 else 1)

    # 对 release 和 artist_name 进行编码
    release_encoder = LabelEncoder()
    new_data['release'] = release_encoder.fit_transform(new_data['release'].values)
    artist_encoder = LabelEncoder()
    new_data['artist_name'] = artist_encoder.fit_transform(new_data['artist_name'].values)

    return new_data


# 歌曲ID和歌曲名称对应关系
def song_titles(data):
    songID_titles = {}
    for index in data.index:
        songID_titles[data.loc[index, 'song']] = data.loc[index, 'title']
    return songID_titles


# ###### Step 2. 排序
# 这里，我们通过svd召回50首歌，然后根据gbdt+lr的结果做权重，给它们做排序，选出其中的5首歌作为推荐结果。
def recommendation(df, user_item, userID):
    recall = recommendation_base_on_MF(df, user_item, userID, 50)
    print('召回完成！')

    # 根据召回的歌曲信息，写出特征向量
    feature_lines = []
    for song in recall.keys():
        feature = new_data[new_data['song'] == song].values[0]
        # 去除rating，将user数值改成当前userID
        feature = feature[:-1]
        feature[0] = userID
        # 存入向量特征中
        feature_lines.append(feature)

    # 利用gbdt+lr计算权重
    weights = lr.predict_proba(onehot.transform(gbdt.apply(feature_lines).reshape(-1, 50)))[:, 1]
    print('权重计算完成！')

    # 计算最终得分
    score = {}
    i = 0
    for song in recall.keys():
        score[song] = recall[song] * weights[i]
        i += 1

    # 选出排名前10的歌曲id
    song_ids = dict(sorted(score.items(), key=lambda x: x[1], reverse=True)[:10])
    # 前10首歌曲的名称
    songID_title = song_titles(data)
    song_topN = [songID_title[i] for i in song_ids.keys()]
    print('最终推荐列表为')

    return song_topN


if __name__ == '__main__':
    data = pd.read_csv('train_dataset.csv')
    new_data = data_process(data)

    # Step 1. GBDT+LR预估
    # 这里，我们做一个ctr点击预估，将点击概率作为权重，与rating结合，作为最终的评分。
    # 为了做这个，我们需要
    # * 分割数据集，一部分作为GBDT的训练集，一部分作为LR的训练集
    # * 先训练GBDT，将其结果作为输入，送进LR里面，再生成结果
    # * 最后看AUC指标
    # 为了加快训练速度，我们从new_data的数据中，取出20%作为训练数据。
    # 取出20%的数据作为数据集
    small_data = new_data.sample(frac=0.2)
    # 将数据集分成gbdt训练街和lr训练集
    X_gbdt, X_lr, y_gbdt, y_lr = train_test_split(small_data.iloc[:, :-1].values, small_data.iloc[:, -1].values, test_size=0.5)

    # 训练gbdt
    gbdt = GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_split=3, min_samples_leaf=2)
    gbdt.fit(X_gbdt, y_gbdt)
    print('当前gbdt训练完成！')

    # one-hot编码
    onehot = OneHotEncoder()
    onehot.fit(gbdt.apply(X_gbdt).reshape(-1, 50))

    # 对gbdt结果进行one-hot编码，然后训练lr
    lr = LogisticRegression(max_iter=500)
    lr.fit(onehot.transform(gbdt.apply(X_lr).reshape(-1, 50)), y_lr)
    print('当前gbdt训练完成！')

    lr_pred = lr.predict(onehot.transform(gbdt.apply(X_lr).reshape(-1, 50)))
    auc_score = roc_auc_score(y_lr, lr_pred)
    print('当前auc为:', auc_score)

    user_item = user_item_matrix(data)

    # 推荐歌曲
    result = recommendation(data, user_item, 100)
    print(result)
