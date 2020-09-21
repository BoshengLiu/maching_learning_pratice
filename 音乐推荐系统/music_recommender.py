import pandas as pd
import numpy as np
import math as mt
import Recommenders
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


# 根据受欢迎程度的歌曲进行推荐
class popularity_recommenders:
    def __init__(self):
        self.train = None
        self.user_id = None
        self.item_id = None
        self.recommendations = None

    def create(self, train, user_id, item_id):
        self.train = train
        self.user_id = user_id
        self.item_id = item_id

        # 对用户id进行聚合操作，得到用户的听歌数量，将听歌数量作为分数
        train_group = train.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_group.rename(columns={user_id: 'score'}, inplace=True)

        # 将数据根据分数和列进行降序操作，当score相同时，根据歌曲id排序
        train_group_sort = train_group.sort_values(['score', self.item_id], ascending=[0, 1])

        # 根据得分生成推荐排名，根据值在原数据中出现的顺序排名
        train_group_sort['rank'] = train_group_sort['score'].rank(ascending=0, method='first')

        # 根据分数推荐前10的歌曲
        self.recommendations = train_group_sort.head(10)

    # 根据用户的id推荐歌曲
    def recommend(self, user_id):
        user_recommendation = self.recommendations

        # 选择需要推荐的用户id
        user_recommendation['user_id'] = user_id

        # 将用户id所在列放在最前列
        cols = user_recommendation.columns.tolist()
        cols[0], cols[-1] = cols[-1], cols[0]
        user_recommendation = user_recommendation[cols]

        return user_recommendation


# 根据歌曲相似度推荐歌曲-协同过滤
class item_similarity_recommendations:
    def __init__(self):
        self.train = None
        self.user_id = None
        self.item_id = None
        self.coordination_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.recommendations = None

    # 获取用户听过的歌曲
    def get_user_items(self, user):
        user_data = self.train[self.train[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items

    # 获取歌曲对应的用户
    def get_item_users(self, item):
        item_data = self.train[self.train[self.item_id] == item]
        item_users = list(item_data[self.user_id].unique())
        return item_users

    # 获取歌曲列表
    def get_items_data(self):
        all_items = list(self.train[self.item_id].unique())
        return all_items

    # 建立用户的协同矩阵
    def construct_coordination_matrix(self, user_songs, all_songs):
        """
        :param user_songs:      用户听过的歌曲（唯一）
        :param all_songs:       歌曲库总歌数
        :return: coordination_matrix    协同矩阵（user_songs X all_songs）
        """
        ##################################
        # 现在要计算的是选中的测试用户推荐什么歌曲，根据听过听歌人的交集与并集情况来计算
        # 流程如下：
        # 1. 先把选中测试用户所听过的歌曲全部拿到
        # 2. 找出这些歌曲中每一个歌曲都被哪些用户听过
        # 3. 在整个歌曲中遍历每一首歌曲，计算它与选中测试用户中每一个听过歌曲的 Jaccard 相似系数
        ##################################

        # 测试用户听过的歌被其他用户听过，将这些用户提取出来
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        # 初始化矩阵（用户听过的歌曲数 X 歌曲库总数）
        coordination_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        # 遍历歌曲库
        for i in range(len(all_songs)):
            songs_i = self.train[self.train[self.item_id] == all_songs[i]]  # 播放过第i首歌曲的数据集
            users_i = set(songs_i[self.user_id].unique())  # 听过第i首歌曲的所有用户

            # 遍历当前用户听过的歌曲
            for j in range(0, len(user_songs)):
                users_j = user_songs_users[j]  # 在测试用户听过的歌曲中，听过第j首歌曲的用户
                users_intersection = users_i.intersection(users_j)  # 计算'听过第i首歌曲的用户'和'听过第j首歌曲的用户'的交集

                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)  # 计算'听过第i首歌曲的用户'和'听过第j首歌曲的用户'的并集

                    # 对应位置更新数据，通过 Jaccard 相似系数来进行衡量
                    coordination_matrix[j, i] = float(len(users_intersection) / len(users_union))

                else:
                    coordination_matrix[j, i] = 0
        return coordination_matrix

    # 使用协同矩阵得出最佳推荐结果
    def generate_top_recommendation(self, user, coo_matrix, all_songs, user_songs):
        print("协同矩阵中的非零值: %d" % np.count_nonzero(coo_matrix))

        # 计算用户所有歌曲在协同矩阵中得分的加权平均值
        user_sim_scores = coo_matrix.sum(axis=0) / float(coo_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores).tolist()

        # 根据用户的得分进行倒序排列，同时将分数和排名对应
        # sort_index ==> [(score_1,rank_1),(score_2,rank_2),...]
        sort_index = sorted(((e, i) for i, e in enumerate(user_sim_scores)), reverse=True)

        # 建立 dataframe
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)

        # 在 dataframe 中填入10个基于推荐系统推荐的结果
        rank = 1
        for i in range(0, len(sort_index)):  # 遍历 sort_index
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, all_songs[sort_index][i][1], sort_index[i][0], rank]   # 给每行赋值
                rank += 1

        # 处理没有推荐的情况
        if df.shape[0] == 0:
            print("目前用户没有歌曲用于训练基于项目相似度的推荐模型.")
            return -1
        else:
            return df

    # 建立基于项目相似度的推荐系统模型
    def create(self, train, user_id, item_id):
        self.train = train
        self.user_id = user_id
        self.item_id = item_id

    # 使用基于项目相似度的推荐系统模型
    def recommend(self, user):
        # 每个用户听过的歌曲数，不包含重复值
        user_songs = self.get_item_users(user)
        print("用户听过的歌曲数：%d" % len(user_songs))

        # 获取歌曲的集合
        all_songs = self.get_items_data()
        print("训练集包含的歌曲有：%d 首." % len(all_songs))

        # 建立协同矩阵
        coordination_matrix = self.construct_coordination_matrix(user_songs, all_songs)

        # 使用协同矩阵建立推荐系统
        df_recommendation = self.generate_top_recommendation(user, coordination_matrix, all_songs, user_songs)

        return df_recommendation

    # 根据歌曲相似度推荐歌曲
    def get_similar_items(self, item_list):
        # 用户听过的歌曲数
        user_songs = item_list

        # 获取歌曲的集合
        all_songs = self.get_items_data()
        print("训练集包含的歌曲有：%d 首." % len(all_songs))

        # 建立协同矩阵
        coordination_matrix = self.construct_coordination_matrix(user_songs, all_songs)

        # 使用协同矩阵建立推荐系统
        user = ""
        df_recommendation = self.generate_top_recommendation(user, coordination_matrix, all_songs, user_songs)

        return df_recommendation


if __name__ == '__main__':
    user_count_df = pd.read_csv('user_playcount_df.csv')
    song_count_df = pd.read_csv('song_playcount_df.csv')
    df_sub = pd.read_csv('triplet_dataset_sub_song_merged.csv')

