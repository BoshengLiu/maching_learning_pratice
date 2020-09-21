import pandas as pd
import numpy as np
import math as mt
import Recommenders
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix



# 基于歌曲相似度推荐-协同过滤
def similarity_recommendation(user_num, triplet_dataset_merged_df):
    song_count_subset = song_count_df.head(n=5000)
    song_subset = list(song_count_subset.song)
    triplet_dataset_sub_song_merged_sub = triplet_dataset_merged_df[triplet_dataset_merged_df.song.isin(song_subset)]

    train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size=0.30, random_state=0)
    is_model = Recommenders.item_similarity_recommender_py()
    is_model.create(train_data, 'user', 'title')

    # 选择一个推荐的对象-第 user_num 个人
    user_id = list(train_data.user)[user_num]
    # 获取该对象所听的歌曲数
    # user_items = is_model.get_user_items(user_id)
    print(is_model.recommend(user_id))


# 基于矩阵分解(SVD)的推荐
def matrix_recommendations(triplet_dataset_merged_df):
    # 先计算歌曲被当前用户播放量/用户播放总量当做分值
    triplet_dataset_sub_song_merged_sum_df = triplet_dataset_merged_df[['user', 'listen_count']].groupby('user').sum().reset_index()
    triplet_dataset_sub_song_merged_sum_df.rename(columns={'listen_count': 'total_listen_count'}, inplace=True)
    triplet_dataset_merged_df = pd.merge(triplet_dataset_merged_df, triplet_dataset_sub_song_merged_sum_df)
    triplet_dataset_merged_df['fractional_play_count'] = triplet_dataset_merged_df['listen_count'] / triplet_dataset_merged_df[
        'total_listen_count']
    # 取一个例子
    sample = triplet_dataset_merged_df[triplet_dataset_merged_df.user == 'd6589314c0a9bcbca4fee0c93b14bc402363afea'][
        ['user', 'song', 'listen_count', 'fractional_play_count']].head()
    print(sample)

    # 对数据进行编码处理-对用户id和歌曲id建立索引
    small_set = triplet_dataset_merged_df
    user_codes = small_set.user.drop_duplicates().reset_index()  # 去除用户中的重复项
    song_codes = small_set.song.drop_duplicates().reset_index()  # 去除歌曲中的重复项
    user_codes.rename(columns={'index': 'user_index'}, inplace=True)  # 重新命名用户索引
    song_codes.rename(columns={'index': 'song_index'}, inplace=True)  # 重新命名歌曲索引
    song_codes['so_index_value'] = list(song_codes.index)  # 建立歌曲索引
    user_codes['us_index_value'] = list(user_codes.index)  # 建立用户索引
    small_set = pd.merge(small_set, song_codes, how='left')  # 将歌曲索引合并到数据集中
    small_set = pd.merge(small_set, user_codes, how='left')  # 将用户索引合并到数据集中
    mat_candidate = small_set[['us_index_value', 'so_index_value', 'fractional_play_count']]
    data_array = mat_candidate.fractional_play_count.values
    row_array = mat_candidate.us_index_value.values
    col_array = mat_candidate.so_index_value.values

    data_sparse = coo_matrix((data_array, (row_array, col_array)), dtype=float)  # 构造矩阵
    return small_set, data_sparse


# 计算 SVD 矩阵
def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)
    print("U-shape", U.shape)
    print("s-shape", s.shape)
    print("V-shape", Vt.shape)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)  # 将特征值转换为对角矩阵
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)  # 将 U,S,V 转换为稀疏矩阵
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


# 计算推荐过程
def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S * Vt
    max_recommendation = 250  # 最大推荐次数
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID, max_recommendation), dtype=np.float16)  # 推荐矩阵大小
    for userTest in uTest:
        print('U[userTest, :', U[userTest, :].shape)  # 获得当前测试的特征
        prod = U[userTest, :] * rightTerm  # 得到当前用户对所有歌曲的特征结果
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings


if __name__ == '__main__':
    user_count_df = pd.read_csv('user_playcount_df.csv')
    song_count_df = pd.read_csv('song_playcount_df.csv')
    triplet_dataset_df = pd.read_csv('triplet_dataset_sub_song.csv')
    track_metadata_df = pd.read_csv('track_metadata_df_sub.csv')
    triplet_dataset_merged_df = pd.read_csv('triplet_dataset_sub_song_merged.csv')

    # 简单暴力，排行榜单推荐播放次数前20的歌曲
    # recommender = popularity_recommenders()
    # recommender.recommendations()

    # 基于歌曲相似度推荐
    # similarity_recommendation()

    # 基于 SVD 矩阵分解的推荐系统
    # K = 50  # 设置特征值个数
    # small_set = matrix_recommendations(triplet_dataset_merged_df)[0]
    # urm = matrix_recommendations(triplet_dataset_merged_df)[1]
    # MAX_PID = urm.shape[1]
    # MAX_UID = urm.shape[0]
    #
    # U, S, Vt = compute_svd(urm, K)  # 计算 SVD 矩阵
    #
    # uTest = [4, 5, 6, 7, 8, 873, 23]  # 推荐的用户索引
    # uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)
    # for user in uTest:  # 打印用户推荐的结果
    #     print("Recommendation for user with user id {}".format(user))
    #     rank_value = 1
    #     for i in uTest_recommended_items[user, 0:10]:
    #         song_details = small_set[small_set.so_index_value == i].drop_duplicates('so_index_value')[['title', 'artist_name']]
    #         print("The number {} recommended song is {} BY {}".format(rank_value, list(song_details['title'])[0], list(song_details['artist_name'])[0]))
    #         rank_value += 1
