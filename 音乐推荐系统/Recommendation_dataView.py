import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def user_playcount_distribution(df, limit_num):
    print(df['play_count'].describe())
    all_playcount_num = np.sum(df['play_count'])
    user_playcount_num = df[df['play_count'].values > limit_num]['play_count'].sum()

    all_user_num = len(df)
    up_user_num = len(df[df['play_count'].values > limit_num])

    print("听歌数量大于100的用户占总用户的比例为：", (str(round(up_user_num * 100 / all_user_num, 4)) + '%'))
    print("听歌数量大于100的用户产生的播放量占总体播放量的比例为：", (str(round(user_playcount_num * 100 / all_playcount_num, 4)) + '%'))

    plt.hist(df['play_count'].values, bins=5000, facecolor='green', alpha=0.5)
    plt.xlabel('Play Counts')
    plt.ylabel('Num of Users')
    plt.title(r'$\mathrm{Histogram\ of\ User\ Play\ Count\ Distribution}\ $')
    plt.grid(True)
    plt.show()


def song_playcount_distribution(df, limit_num):
    print(df['play_count'].describe())
    all_playcount_num = np.sum(df['play_count'])
    song_playcount_num = df[df['play_count'].values > limit_num]['play_count'].sum()

    all_song_num = len(df)
    up_song_num = len(df[df['play_count'].values > limit_num])

    print("播放次数大于100的歌曲数量占歌库的比例为：", (str(round(up_song_num * 100 / all_song_num, 4)) + '%'))
    print("播放次数大于100的歌曲产生的播放量占总播放量的比例为：", (str(round(song_playcount_num * 100 / all_playcount_num, 4)) + '%'))

    plt.hist(df['play_count'].values, bins=5000, facecolor='green', alpha=0.5)
    plt.xlabel('Play Counts')
    plt.ylabel('Num of Songs')
    plt.title(r'$\mathrm{Histogram\ of\ Songs\ Play\ Count\ Distribution}\ $')
    plt.grid(True)
    plt.show()


def most_popular_songs(df):
    popular_songs = df[['title', 'play_count']].groupby('title').sum().reset_index()
    popular_songs_top_20 = popular_songs.sort_values('play_count', ascending=False).head(n=20)
    print("The most 20 popular songs is:")
    print(popular_songs_top_20)

    song_playcounts = {}
    for song, counts in zip(popular_songs_top_20['title'], popular_songs_top_20['play_count']):
        song_playcounts[song] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(song_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def most_popular_release(df):
    popular_release = df[['release', 'play_count']].groupby('release').sum().reset_index()
    popular_release_top_20 = popular_release.sort_values('play_count', ascending=False).head(n=20)
    print("The most 20 popular release is:")
    print(popular_release_top_20)

    release_playcounts = {}
    for release, counts in zip(popular_release_top_20['release'], popular_release_top_20['play_count']):
        release_playcounts[release] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(release_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def most_popular_artist(df):
    popular_artist = df[['artist_name', 'play_count']].groupby('artist_name').sum().reset_index()
    popular_artist_top_20 = popular_artist.sort_values('play_count', ascending=False).head(n=20)
    print("The most 20 popular artists is:")
    print(popular_artist_top_20)

    artist_playcounts = {}
    for artist, counts in zip(popular_artist_top_20['artist_name'], popular_artist_top_20['play_count']):
        artist_playcounts[artist] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(artist_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    user_play_count = pd.read_csv('user_df.csv')
    song_play_count = pd.read_csv('song_df.csv')
    train_df = pd.read_csv('train_dataset.csv')

    # 1. 分别查看用户听歌数目分布、歌曲被播放的分布
    user_playcount_distribution(user_play_count, limit_num=100)
    song_playcount_distribution(song_play_count, limit_num=100)


    # 2. 分别查看最受欢迎的歌手、最受欢迎的的专辑、最受欢迎的歌手
    most_popular_songs(train_df)
    most_popular_release(train_df)
    most_popular_artist(train_df)
