import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 最流行的歌曲 top20
def most_popular_songs(df):
    popular_songs = df[['title', 'listen_count']].groupby('title').sum().reset_index()
    popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular songs is:")
    print(popular_songs_top_20)  # 最流行的前20首歌曲

    objects = (list(popular_songs_top_20['title']))
    y_pos = np.arange(len(objects))
    performance = list(popular_songs_top_20['listen_count'])

    plt.figure(figsize=(12, 8))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Item count')
    plt.title('Most popular songs')
    plt.show()


# 最受欢迎的唱片 top20
def most_popular_release(df):
    popular_release = df[['release', 'listen_count']].groupby('release').sum().reset_index()
    popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular release is:")
    print(popular_release_top_20)

    objects = (list(popular_release_top_20['release']))
    y_pos = np.arange(len(objects))
    performance = list(popular_release_top_20['listen_count'])

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Item count')
    plt.title('Most popular Release')
    plt.show()


# 最受欢迎的歌手 top20
def most_popular_artist(df):
    popular_artist = df[['artist_name', 'listen_count']].groupby('artist_name').sum().reset_index()
    popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular artists is:")
    print(popular_artist_top_20)

    objects = (list(popular_artist_top_20['artist_name']))
    y_pos = np.arange(len(objects))
    performance = list(popular_artist_top_20['listen_count'])

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Item count')
    plt.title('Most 20 popular Artists')
    plt.show()


# 歌曲数量分布
def song_count_distribution(df):
    user_song_count_distribution = df[['user', 'title']].groupby('user').count().reset_index().sort_values(
        by='title', ascending=False)
    user_song_count_distribution.title.describe()
    x = user_song_count_distribution.title
    plt.hist(x, 50, facecolor='green', alpha=0.75)
    plt.xlabel('Play Counts')
    plt.ylabel('Num of Users')
    plt.title(r'$\mathrm{Histogram\ of\ User\ Play\ Count\ Distribution}\ $')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('triplet_dataset_sub_song_merged.csv')
    most_popular_songs(df)  # 最受欢迎的歌手
    most_popular_release(df)  # 最受欢迎的的专辑
    most_popular_artist(df)  # 最受欢迎的歌手
    song_count_distribution(df)  # 歌曲数量分布
