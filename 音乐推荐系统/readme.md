# 从零开始搭建音乐推荐系统
在本篇博客中，我们将从0搭建一个音乐推荐系统，其中的流程也可以用来搭建其他内容的推荐系统。我们将整个过程分为三个部分，分别是：数据预处理、召回、排序。

## 一. 数据预处理
### 1. 初步提取txt文件
* 用户的播放记录数据集train_triplets.txt格式是这样的：用户、歌曲、播放次数。
```python
df = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])
print(df.head())
```

![](https://upload-images.jianshu.io/upload_images/16911112-f14b1e97bb885e3c.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


* 然后对数据进行聚合处理，得到每位用户的歌曲播放次数、每首歌曲的播放次数，参考程序如下：
```python
def play_count(num, key_name):
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
    
for i, j in zip([0, 1], ['user', 'song']):
    play_count(i, j)

```

### 2. 可视化数据

* 对数据进行可视化操作，观察用户听过的歌曲数，如下图所示：

![](https://upload-images.jianshu.io/upload_images/16911112-4df23a5ad3f695d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过上图可以发现有一大部分用户的歌曲播放量少于100，少于100的歌曲播放量在持续几年的时间长度上来看是不正常的。造成这种现象的原因，可能是这些用户不喜欢听歌，只是偶尔点开。对于这些用户，我们看看他们在总体数据上的占比情况。

* 筛选数据，分别获取听歌次数多的用户和播放次数多的歌曲，参考程序如下：
```python
df = pd.read_csv('user_playcount_df.csv')

all_playcount_num = np.sum(df['play_count'])
user_playcount_num = df[df['play_count'].values > 100]['play_count'].sum()

all_user_num = len(df)
up_user_num = len(df[df['play_count'].values > 100])

print("听歌数量大于100的用户占总用户的比例为：", (str(round(up_user_num * 100 / all_user_num, 4)) + '%'))
print("听歌数量大于100的用户产生的播放量占总体播放量的比例为：", (str(round(user_playcount_num * 100 / all_playcount_num, 4)) + '%'))
```

![](https://upload-images.jianshu.io/upload_images/16911112-97c5dc2cdf32fac3.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过图片可以看出，39.40%的用户听了80.02%的歌曲，那么可以舍去听歌少的用户，因为用户基本上不怎么听歌，或者偶尔听听。

* 同理，观察下图，29.18%的歌曲播放量占了总体播放量的95.17%，可以舍去播放量少的歌曲，可以确定播放次数少的歌曲基本上没什么人听。

![](https://upload-images.jianshu.io/upload_images/16911112-c099c0539c0e2248.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/16911112-792d4bea502ce9e5.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 3. 处理txt数据
* 根据前面可视化，保留听歌次数大于100的用户和播放次数大于100的歌曲，然后根据筛选要求提取txt文件，参考程序如下：
```python
def triplet_dataset_sub():
    # 筛选数据，保留播放次数大于100的歌曲/听歌数量大于100的用户
    user_df = pd.read_csv("user_df.csv")
    song_df = pd.read_csv("song_df.csv")

    user_subset = list(user_df['user'])
    song_subset = list(song_df['song'])

    triplet_dataset = pd.read_csv('train_triplets.txt', sep='\t', header=None, names=['user', 'song', 'play_count'])
    triplet_dataset_sub = triplet_dataset[triplet_dataset['user'].isin(user_subset)]
    triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub['song'].isin(song_subset)]
    triplet_dataset_sub_song.to_csv('triplet_dataset_sub.csv', index=False)

```

### 4. 提取db文件
* 读取数据
```python
conn = sqlite3.connect('track_metadata.db')
cur = conn.cursor()
cur.execute("select * from sqlite_master where type='table'")
cur.fetchall()

```

* 提取文件
```python
song_df = pd.read_csv("song_df.csv")
song_subset = list(song_df['song']) 
track_metadata_df = pd.read_sql(con=conn, sql='select * from songs')
track_metadata_df_sub = track_metadata_df[track_metadata_df['song_id'].isin(song_subset)]
track_metadata_df_sub.to_csv('track_metadata_df.csv', index=False)

```

### 5. 合并数据集，清理无用数据集
* 首先去除重复的数据
```python
triplet_dataset_sub_song = pd.read_csv('triplet_dataset_sub.csv')
track_metadata_df_sub = pd.read_csv('track_metadata_df_sub.csv')

# 去除重复数据
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
```

* 合并数据集并重命名
```python
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')
triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True)

```

* 清洗掉无用数据
```python
clean_cols = ['song_id', 'artist_id', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'track_7digitalid',
                  'shs_perf', 'shs_work', 'artist_mbid', 'track_id']
final_dataset.drop(columns=clean_cols, inplace=True)

```

* 对歌曲id和用户id进行编码
```python
user_encoder = LabelEncoder()
final_dataset['user'] = user_encoder.fit_transform(final_dataset['user'].values)
song_encoder = LabelEncoder()
final_dataset['song'] = song_encoder.fit_transform(final_dataset['song'].values)

```
    
* 对数据类型进行转换
```python
cols = ['user', 'song', 'listen_count', 'year']
for i in cols:
    final_dataset[i] = final_dataset[i].astype('int32')

```

### 6. 最终数据可视化
* 最受欢迎的歌曲top20

```python
def most_popular_songs(df):
    popular_songs = df[['title', 'listen_count']].groupby('title').sum().reset_index()
    popular_songs_top_20 = popular_songs.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular songs is:")
    print(popular_songs_top_20)

    song_playcounts = {}
    for song, counts in zip(popular_songs_top_20['title'], popular_songs_top_20['listen_count']):
        song_playcounts[song] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(song_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

```

![](https://upload-images.jianshu.io/upload_images/16911112-2ade8997e70783e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 最受欢迎的唱片top20
```python
def most_popular_release(df):
    popular_release = df[['release', 'listen_count']].groupby('release').sum().reset_index()
    popular_release_top_20 = popular_release.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular release is:")
    print(popular_release_top_20)

    release_playcounts = {}
    for release, counts in zip(popular_release_top_20['release'], popular_release_top_20['listen_count']):
        release_playcounts[release] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(release_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

```

![](https://upload-images.jianshu.io/upload_images/16911112-170322f601d1204e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 最受欢迎的歌手top20

```python
def most_popular_artist(df):
    popular_artist = df[['artist_name', 'listen_count']].groupby('artist_name').sum().reset_index()
    popular_artist_top_20 = popular_artist.sort_values('listen_count', ascending=False).head(n=20)
    print("The most 20 popular artists is:")
    print(popular_artist_top_20)

    artist_playcounts = {}
    for artist, counts in zip(popular_artist_top_20['artist_name'], popular_artist_top_20['listen_count']):
        artist_playcounts[artist] = counts

    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1000, height=800)
    wc.generate_from_frequencies(artist_playcounts)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

```
![](https://upload-images.jianshu.io/upload_images/16911112-6cea81d6e0ef7ed3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## 二. 召回



---

## 三、排序


