# 从零开始搭建音乐推荐系统
在本篇博客中，我们将从0搭建一个音乐推荐系统，其中的流程也可以用来搭建其他内容的推荐系统。我们将整个过程分为三个部分，分别是：数据预处理、召回、排序。

# 一. 数据预处理
## 1. 初步提取txt文件
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

## 2. 可视化数据

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

## 3. 处理txt数据
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

## 4. 提取db文件
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

## 5. 合并数据集，清理无用数据集
* 首先去除重复的数据
```python
triplet_dataset_sub_song = pd.read_csv('triplet_dataset_sub.csv')
track_metadata_df_sub = pd.read_csv('track_metadata_df_sub.csv')

# 去除重复数据
track_metadata_df_sub = track_metadata_df_sub.drop_duplicates(['song_id'])
```

* 合并数据集
```python
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left', left_on='song', right_on='song_id')

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

## 6. 最终数据可视化
* 最受欢迎的歌曲top20

```python
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

```

![](https://upload-images.jianshu.io/upload_images/16911112-2ade8997e70783e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 最受欢迎的唱片top20
```python
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

```

![](https://upload-images.jianshu.io/upload_images/16911112-170322f601d1204e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 最受欢迎的歌手top20

```python
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

```
![](https://upload-images.jianshu.io/upload_images/16911112-6cea81d6e0ef7ed3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

# 二. 召回
* 对于系统的召回阶段，我们将给出如下三种推荐方式，分别是：
    * 基于排行榜的推荐
    
    * 基于协同过滤的推荐
    * 基于矩阵分解的推荐

## 1. 基于排行榜的推荐
* 我们将每首歌听过的人数作为每首歌的打分，这里之所以不将点击量作为打分，是因为一个人可能对一首歌多次点击，但这首歌其他人并不喜欢。
```python
def most_popularity_songs(train_data, N):
    df_popular = train_data.groupby(['song', 'title']).agg({'user': 'count'}).reset_index()
    df_popular = df_popular.sort_values(by='user', ascending=False)
    return df_popular.head(N)

most_popularity_songs(train_df, 20)
```

## 2. 基于协同过滤的推荐
* 协同过滤需要用户-物品评分矩阵，用户对某首歌的评分的计算公式如下，
    * 该用户的最大歌曲点击量
    
    * 当前歌曲点击量/最大歌曲点击量
    * 评分为log(2 + 上述比值)

* 得到用户-物品评分矩阵之后，我们用 surprise 库中的 knnbasic函数进行协同过滤。

```python
# 每首歌的总点击量
df = train_data[['song', 'play_count']].groupby('song').sum()
df.rename(columns={'play_count': 'all_counts'}, inplace=True)

train_df = train_data.merge(df, left_on='song', right_on='song')
train_df['rating'] = np.log(train_df['play_count'].values / train_df['all_counts'] + 2)

# 得到用户-音乐评分矩阵
user_item = train_df[['user', 'song', 'rating']]
user_item = user_item.rename(columns={'song': 'item'})

```

### 2.1 基于物品协同过滤的推荐

```python
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

```


### 2.2 基于用户协同过滤的推荐

```python
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

```


## 3. 基于矩阵分解的推荐

```python
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

```

---

## 三、排序
### 1. GBDT+LR预估
* 这里，我们做一个ctr点击预估，将点击概率作为权重，与rating结合，作为最终的评分。

```python
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

```

### 2. 排序
* 这里，我们通过svd召回50首歌，然后根据gbdt+lr的结果做权重，给它们做排序，选出其中的5首歌作为推荐结果。

```python
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

```

### 3. 结余
* 限于机器性能和时间所限，不能训练更多的数据，显然是未来可以提高的部分。在排序阶段，我们还可以用深度学习的相关算法，效果可能也不错。如果有更多的数据，比如像大众点评的结果查询结果，
我们或许还可以做重排序。