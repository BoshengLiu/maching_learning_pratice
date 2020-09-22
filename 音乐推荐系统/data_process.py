import pandas as pd
import sqlite3
from tqdm import *
from sklearn.preprocessing import LabelEncoder


def play_count_df(num, key_name):
    output_dict = {}
    with open('train_triplets.txt') as f:
        for line_num, line in tqdm(enumerate(f)):
            data = line.split('\t')[num]
            play_count = int(line.split('\t')[2])
            if data in output_dict:
                play_count += output_dict[data]
                output_dict.update({data: play_count})
            output_dict.update({data: play_count})
    output_list = [{key_name: k, 'play_count': v} for k, v in output_dict.items()]
    output_df = pd.DataFrame(output_list)
    output_df = output_df.sort_values(by='play_count', ascending=False)
    output_df.to_csv(key_name + "_playcount_df.csv", index=False)


def triplet_dataset():
    user_df = pd.read_csv("user_df.csv")
    song_df = pd.read_csv("song_df.csv")

    user_subset = list(user_df['user'])
    song_subset = list(song_df['song'])

    triplet_dataset = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])

    triplet_dataset_sub = triplet_dataset[triplet_dataset['user'].isin(user_subset)]
    triplet_dataset = triplet_dataset_sub[triplet_dataset_sub['song'].isin(song_subset)]
    triplet_dataset.to_csv('triplet_dataset.csv', index=False)


def extract_database():
    song_df = pd.read_csv("song_df.csv")
    song_subset = list(song_df['song'])

    # 连接数据库，提取表
    conn = sqlite3.connect('track_metadata.db')
    cur = conn.cursor()
    cur.execute("select * from sqlite_master where type='table'")
    cur.fetchall()

    # 读取sql文件，根据数据集筛选数据
    track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
    track_metadata_df_sub = track_metadata_df[track_metadata_df['song_id'].isin(song_subset)]
    track_metadata_df_sub.to_csv('track_metadata_df.csv', index=False)


def data_clean():
    triplet_dataset_sub_song = pd.read_csv('triplet_dataset.csv')
    track_metadata_df_sub = pd.read_csv('track_metadata_df.csv')

    # 去除重复数据
    track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])

    # 合并数据并重命名
    final_dataset = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
    final_dataset.rename(columns={'play_count': 'listen_count'}, inplace=True)

    # 清洗掉无用数据
    clean_cols = ['song_id', 'artist_id', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'track_7digitalid',
                  'shs_perf', 'shs_work', 'artist_mbid', 'track_id']
    for i in clean_cols:
        del final_dataset[i]

    # 对歌曲id和用户id进行编码
    user_encoder = LabelEncoder()
    final_dataset['user'] = user_encoder.fit_transform(final_dataset['user'].values)
    song_encoder = LabelEncoder()
    final_dataset['song'] = song_encoder.fit_transform(final_dataset['song'].values)

    # 数据类型转换
    cols = ['user', 'song', 'listen_count', 'year']
    for i in cols:
        final_dataset[i] = final_dataset[i].astype('int32')

    final_dataset.to_csv('train_dataset.csv', index=False)


if __name__ == '__main__':
    ##############################################################
    # 将txt文件提取出用户、听歌次数和歌曲、歌曲播放次数
    ##############################################################
    play_count_df(0, 'user')
    play_count_df(1, 'song')

    ##############################################################
    # 根据可视化结果保留播放次数大于100的歌曲/听歌数量大于100的用户
    ##############################################################
    user_play_count = pd.read_csv('user_playcount_df.csv')
    song_play_count = pd.read_csv('song_playcount_df.csv')
    user_subset = user_play_count[user_play_count['play_count'] > 100]
    song_subset = song_play_count[song_play_count['play_count'] > 100]
    user_subset.to_csv("user_df.csv", index=False)
    song_subset.to_csv("song_df.csv", index=False)

    ##############################################################
    # 根据可视化结果提取txt文件
    ##############################################################
    triplet_dataset()

    ##############################################################
    # 提取db文件
    ##############################################################
    extract_database()

    ##############################################################
    # 合并数据，清理无用数据，得到最后的训练数据
    ##############################################################
    data_clean()
