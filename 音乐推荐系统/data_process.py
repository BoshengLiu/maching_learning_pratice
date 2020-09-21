import pandas as pd
import sqlite3


# 将 .txt文件提取出用户、听歌次数和歌曲、歌曲播放次数
def play_count(num, key_name):
    output_dict = {}
    with open('train_triplets.txt') as f:
        for line_num, line in enumerate(f):
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


# 将 .txt文件提取出数据用户、歌曲、播放次数，然后进行筛选
def triplet_dataset_sub_song():
    play_count_df = pd.read_csv('user_playcount_df.csv')
    song_count_df = pd.read_csv('song_playcount_df.csv')
    # 筛选数据，选取前100000数据对应的用户、歌曲、播放次数
    # 这里也可以自定义数据（比如保留播放次数大于100的歌曲/听歌数量大于100的用户）
    # play_count_subset = play_count_df[play_count_df['play_count'] > 100]
    # song_count_subset = song_count_df[song_count_df['play_count'] > 100]
    play_count_subset = play_count_df.head(n=100000)
    song_count_subset = song_count_df.head(n=100000)
    user_subset = list(play_count_subset.user)
    song_subset = list(song_count_subset.song)

    # 读取 .txt 文件并对应列名
    triplet_dataset = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])
    # 筛选数据，保留前100000数据对应的用户、歌曲、播放次数，保存为 .csv 文件
    triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
    triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
    triplet_dataset_sub_song.to_csv('triplet_dataset_sub_song.csv', index=False)


# 提取 .db文件-存放歌曲信息
def extract_database():
    song_count_df = pd.read_csv('song_playcount_df.csv')
    song_count_subset = song_count_df.head(n=100000)
    song_subset = list(song_count_subset.song)

    # 连接数据库，提取表
    conn = sqlite3.connect('track_metadata.db')
    cur = conn.cursor()
    cur.execute("select * from sqlite_master where type='table'")
    # 获取所有表的表名
    # sql = "select name from sqlite_master where type='table' order by name"
    cur.fetchall()

    # 读取 sql文件，筛选出歌曲 id 存在于播放次数前100000的数据，保存为 .csv 文件
    track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
    track_metadata_df_sub = track_metadata_df[track_metadata_df.song_id.isin(song_subset)]
    track_metadata_df_sub.to_csv('track_metadata_df_sub.csv', index=False)


# 合并数据(用户播放信息与歌曲信息)，清理无用数据
def data_clean():
    triplet_dataset_sub_song = pd.read_csv('triplet_dataset_sub_song.csv')
    track_metadata_df_sub = pd.read_csv('track_metadata_df_sub.csv')

    # 去除重复数据
    track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])

    # 合并数据并重命名
    triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
    triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True)

    # 清洗掉无用数据
    clean_cols = ['song_id', 'artist_id', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'track_7digitalid',
                  'shs_perf', 'shs_work', 'artist_mbid', 'track_id']
    for i in clean_cols:
        del (triplet_dataset_sub_song_merged[i])

    triplet_dataset_sub_song_merged.to_csv('triplet_dataset_sub_song_merged.csv', index=False)


if __name__ == '__main__':
    # 将 .txt文件提取出用户、听歌次数和歌曲、歌曲播放次数
    # for i, j in zip([0, 1], ['user', 'song']):
    #     play_count(i, j)

    # triplet_dataset_sub_song()  # 将 .txt文件提取出用户、歌曲、播放次数

    # extract_database()  # 提取 .db文件

    data_clean()  # 合并数据，清理无用数据，得到最后的训练数据
