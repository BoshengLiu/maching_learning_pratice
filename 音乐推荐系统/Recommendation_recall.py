import pandas as pd
import numpy as np
from surprise import Reader, KNNBasic, SVD, Dataset, accuracy


# 基于排行榜的推荐
def most_popularity_songs(train_data, N):
    # 统计每首歌曲有多少个用户听过
    df_popular = train_data.groupby(['song', 'title']).agg({'user': 'count'}).reset_index()
    df_popular = df_popular.sort_values(by='user', ascending=False)
    return df_popular.head(N)


# 创建用户-物品评分矩阵
def user_item_matrix(train_data):
    # 每首歌的总点击量
    df = train_data[['song', 'play_count']].groupby('song').sum()
    df.rename(columns={'play_count': 'all_counts'}, inplace=True)

    train_df = train_data.merge(df, left_on='song', right_on='song')
    train_df['rating'] = np.log(train_df['play_count'].values / train_df['all_counts'] + 2)

    # 得到用户-音乐评分矩阵
    user_item = train_df[['user', 'song', 'rating']]
    user_item = user_item.rename(columns={'song': 'item'})
    return user_item


# 基于物品相似度的推荐
def recommendation_base_on_itemCF(train_data, user_item_matrix, user_ID, N):
    # 阅读器
    reader = Reader(line_format='user item rating', sep=',')
    # 载入数据
    raw_data = Dataset.load_from_df(user_item_matrix, reader=reader)

    # 构建模型
    raw_data.split(n_folds=5)
    # kf = KFold(n_splits=5)
    knn_item = KNNBasic(k=40, sim_options={'user_based': False})
    # 训练数据，并返回rmse误差
    for train_set, test_set in raw_data.folds():
        knn_item.fit(train_set)
        predictions = knn_item.test(test_set)
        accuracy.rmse(predictions, verbose=True)

    # 用户听过的歌曲合集
    user_songs = {}
    for user, group in user_item_matrix.groupby('user'):
        user_songs[user] = group['item'].values.tolist()
    # 歌曲合集
    songs = user_item_matrix['item'].unique().tolist()
    # 歌曲ID和歌曲名称对应关系
    songID_titles = {}
    for index in train_data.index:
        songID_titles[train_data.loc[index, 'song']] = train_data.loc[index, 'title']

    # itemCF
    # 用户听过的音乐集
    user_items = user_songs[user_ID]

    # 用户对未听过音乐的评分
    item_rating = {}
    for item in songs:
        if item not in user_items:
            item_rating[item] = knn_item.predict(user_ID, item).est

    # 找出评分靠前的N首歌曲
    song_id = dict(sorted(item_rating.items(), key=lambda x: x[1], reverse=True)[:N])
    song_topN = [songID_titles[s] for s in song_id.keys()]

    return song_topN


# 基于用户相似度的推荐
def recommendation_base_on_userCF(train_data, user_item_matrix, user_ID, N):
    # 阅读器
    reader = Reader(line_format='user item rating', sep=',')
    # 载入数据
    raw_data = Dataset.load_from_df(user_item_matrix, reader=reader)

    # 构建模型
    raw_data.split(n_folds=5)
    knn_item = KNNBasic(k=40, sim_options={'user_based': True})
    # 训练数据，并返回rmse误差
    for train_set, test_set in raw_data.folds():
        knn_item.fit(train_set)
        predictions = knn_item.test(test_set)
        accuracy.rmse(predictions, verbose=True)

    # 用户听过的歌曲合集
    user_songs = {}
    for user, group in user_item_matrix.groupby('user'):
        user_songs[user] = group['item'].values.tolist()
    # 歌曲合集
    songs = user_item_matrix['item'].unique().tolist()
    # 歌曲ID和歌曲名称对应关系
    songID_titles = {}
    for index in train_data.index:
        songID_titles[train_data.loc[index, 'song']] = train_data.loc[index, 'title']

    # userCF
    # 用户听过的音乐集
    user_items = user_songs[user_ID]

    # 用户对未听过音乐的评分
    item_rating = {}
    for item in songs:
        if item not in user_items:
            item_rating[item] = knn_item.predict(user_ID, item).est

    # 找出评分靠前的N首歌曲
    song_id = dict(sorted(item_rating.items(), key=lambda x: x[1], reverse=True)[:N])
    song_topN = [songID_titles[s] for s in song_id.keys()]

    return song_topN

# 基于矩阵分解的推荐
def recommendation_base_on_MF(train_data, user_item_matrix, user_ID, N):
    # 阅读器
    reader = Reader(line_format='user item rating', sep=',')
    # 载入数据
    raw_data = Dataset.load_from_df(user_item_matrix, reader=reader)

    # 构建模型
    raw_data.split(n_folds=5)
    algo = SVD(n_factors=40, biased=True)
    # 训练数据，并返回rmse误差
    for train_set, test_set in raw_data.folds():
        algo.fit(train_set)
        predictions = algo.test(test_set)
        accuracy.rmse(predictions, verbose=True)

    # 用户听过的歌曲合集
    user_songs = {}
    for user, group in user_item_matrix.groupby('user'):
        user_songs[user] = group['item'].values.tolist()
    # 歌曲合集
    songs = user_item_matrix['item'].unique().tolist()
    # 歌曲ID和歌曲名称对应关系
    songID_titles = {}
    for index in train_data.index:
        songID_titles[train_data.loc[index, 'song']] = train_data.loc[index, 'title']

    # 基于矩阵分解的推荐
    user_items = user_songs[user_ID]

    # 用户对未听过音乐的评分
    item_rating = {}
    for item in songs:
        if item not in user_items:
            item_rating[item] = algo.predict(user_ID, item).est

    # 找出评分靠前的N首歌曲
    song_id = dict(sorted(item_rating.items(), key=lambda x: x[1], reverse=True)[:N])
    song_topN = song_id

    return song_topN


if __name__ == '__main__':
    train_df = pd.read_csv('train_dataset.csv')

    # 排行榜单推荐播放次数前20的歌曲
    # print(most_popularity_songs(train_df, 20))

    # 创建用户评分矩阵
    users_items = user_item_matrix(train_df)
    # print(users_items)

    # 基于物品协同过滤的推荐
    result = recommendation_base_on_itemCF(train_df, users_items, 1220, 10)
    # print(result)

    # 基于用户协同过滤的推荐
    # result = recommendation_base_on_userCF(train_df, users_items, 1220, 10)
    # print(result)

    # 基于矩阵分解的推荐
    # result = recommendation_base_on_MF(train_df, users_items, 1220, 10)
    # print(result)
