### 赛事题目：[国能日新第二届光伏功率预测赛](https://www.dcjingsai.com/common/cmpt/%E5%9B%BD%E8%83%BD%E6%97%A5%E6%96%B0%E7%AC%AC%E4%BA%8C%E5%B1%8A%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

---

* 目标：利用气象信息、历史数据，通过机器学习、人工智能方法，预测未来电站的发电功率。

* 训练数据集字段：

|  字段  |  备注  |
|--------|----------|
|   时间   |   采集实际功率的时间      |
|   辐照度   |     预测的辐照度     |
|   风速    |   预测的风速   |
|   风向    |   预测的风向   |
|   温度    |   预测的温度   |
|   压强    |   预测的压强   |
|   湿度    |   预测的湿度   |
|   实际辐照度   |   实际采集的辐照度    |
|   实际功率    |   实际采集的辐照度    |

* 测试数据集字段：

|  字段  |  备注  |
|--------|----------|
|   id   |   数据id    |
|   时间   |   采集实际功率的时间      |
|   辐照度   |     预测的辐照度     |
|   风速    |   预测的风速   |
|   风向    |   预测的风向   |
|   温度    |   预测的温度   |
|   压强    |   预测的压强   |
|   湿度    |   预测的湿度   |

* 问题：训练集合测试集有两个字段不同，实际功率是要预测的对象，但是测试集多了一个实际辐照度。两种处理方法：一种是不考虑实际辐照度的影响，直接舍去；另一种方法是将实际辐照度作为第一次训练的预测对象去预测，然后将预测的实际辐照度作为新的特征去加入第二次训练。但是实际上实际辐照度和实际功率的相关性太高，第二种方法会导致气象数据的影响降低。所以这里采取第一种方法，直接舍去实际辐照度。

### 一.数据预处理
---

* 1.先查看数据所包含的内容，是否有缺失值。

![](https://user-gold-cdn.xitu.io/2019/11/7/16e45451da30a3bf?w=1100&h=334&f=png&s=44478)
![](https://user-gold-cdn.xitu.io/2019/11/7/16e45460953c2623?w=342&h=324&f=png&s=28960)

可以看出数据并无缺失值

* 2.查看数据和异常值，进行可视化操作。

![](https://user-gold-cdn.xitu.io/2019/11/6/16e4038901c72fbd?w=1200&h=500&f=png&s=31573)

放大后可以看见详细情况，结果如下：

![](https://user-gold-cdn.xitu.io/2019/11/6/16e4039d12741825?w=1048&h=1804&f=png&s=323325)

发现数据存在异常值，对异常值进行处理，处理完后的可视化结果如下：

![](https://user-gold-cdn.xitu.io/2019/11/6/16e403a5315495dd?w=1200&h=500&f=png&s=44563)

### 二、特征处理
---

* 1.时间处理，这里的时间格式为xxxx-x-x xx:xx:xx，对应年-月-日 小时-分钟-秒，所以我们要将数据拆分成年、月、日、时、分、秒，还可以通过月份划分季节。

* 2.特征差分处理，历史数据也是一个重要的特征，我们对辐照度、温度、湿度、压强进行差分处理，获取他们的差分特征。差分的第一行为空，可以将其填充为0。

* 3.统计特征，原始数据是按照每15分钟统计一次，我们可以统计一小时的数据，观察数据。

* 4.挖掘新特征，这里涉及到了地理学知识，如太阳的高度角、直射纬度等等。

* 5.特征组合，上面对特征进行交叉组合，获取组合特征。

* 6.数据离散化处理，风向属于离散数据，而原始数据为连续值，要进行离散化处理。

* 7.one-hot处理，对一些特征进行one-hot处理，比如季节。

* 8.特征选择，构造了许多新的特征，有些特征可能是无关紧要的，所以需要进行特征选择，过滤掉一些不重要的特征。

### 三、建模
---
特征建立完成，接下来是模型的建立，选择一个合适的模型是很有必要的。可以选择神经网络、XGboost等等，这里我选择了lightGBM。

* 1.参数设定，模型的参数要设置合适的范围，这个需要经验积累。

* 2.自动调参，参数设定完后开始自动调参。可以考虑GridearchCV，但是太耗费时间了，这里我选择了Hyperopt来进行自动调参。参考程序如下：

```js
# lightGBM自动调参
def lgbTraining(x_train, y_train, p):
    train_x, valid_x, train_y, valid_y = train_test_split(x_train.values, y_train.values, test_size=0.3,
                                                          random_state=42)
    train = lgb.Dataset(train_x, train_y)
    valid = lgb.Dataset(valid_x, valid_y, reference=train)

    # 自定义hyperopt的参数空间
    space = {"max_depth": hp.randint("max_depth", 15),
             "num_trees": hp.randint("num_trees", 20),
             'learning_rate': hp.randint('learning_rate', 20),
             "num_leaves": hp.randint("num_leaves", 10),
             "lambda_l1": hp.randint("lambda_l1", 6)
             }

    def argsDict_tranform(argsDict, isPrint=False):
        argsDict["max_depth"] = argsDict["max_depth"] + 10
        argsDict["num_trees"] = argsDict["num_trees"] * 5 + 100
        argsDict["learning_rate"] = argsDict["learning_rate"] * 0.01 + 0.01
        argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
        argsDict["lambda_l1"] = argsDict["lambda_l1"] * 0.1
        if isPrint:
            print(argsDict)
        else:
            pass
        return argsDict

    def lightgbm_factory(argsDict):
        argsDict = argsDict_tranform(argsDict)
        params = {'nthread': -1,  # 进程数
                  'max_depth': argsDict['max_depth'],  # 最大深度
                  'num_trees': argsDict['num_trees'],  # 树的数量
                  'learning_rate': argsDict['learning_rate'],  # 学习率
                  'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
                  'lambda_l1': argsDict["lambda_l1"],  # L1 正则化
                  'lambda_l2': 0,  # L2 正则化
                  'objective': 'regression',
                  'bagging_seed': 100  # 随机种子,light中默认为100
                  }
        params['metric'] = ['mae']
        model_lgb = lgb.train(params, train, num_boost_round=20000, valid_sets=[valid], early_stopping_rounds=100)
        return get_transformer_score(model_lgb)

    # 获取实际功率大于0.03*p的部分
    valid_y_new = valid_y[valid_y > 0.03 * p]
    valid_y_new_index = np.argwhere(valid_y > 0.03 * p)

    def get_transformer_score(transformer):
        model = transformer
        prediction = model.predict(valid_x, num_iteration=model.best_iteration)
        prediction_new = prediction[valid_y_new_index]
        return mean_absolute_error(valid_y_new, prediction_new)

    # 开始使用hyperopt进行自动调参
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(lightgbm_factory, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None)
    MAE = lightgbm_factory(best) / p

    return MAE, best
```

* 3.模型检验，将模型得到的参数去预测测试集，然后上线获取分数。

* 4.参数优化，如果线下模型表现的比线上好，则是过拟合，需要减少过拟合。

* 这里附上分数和排名，排名不是很高，特征工程做得不是很好，需要多多学习特征工程的构建。同时附上rank25的baseline，好好学习一下。
![](https://upload-images.jianshu.io/upload_images/16911112-25ecc569bed409db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* 参考了baseline修改后的分数有了提高，但是还有待优化。
![](https://upload-images.jianshu.io/upload_images/16911112-a479252e3f1a5d32.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
