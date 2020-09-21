### 赛事题目：[商家客户购买转化率预测](https://www.dcjingsai.com/common/cmpt/2019%E6%95%B0%E6%8D%AE%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

* 内容：购买转化率是品牌商家在电商平台运营时最关注的指标之一，本次大赛中云积互动提供了品牌商家的历史订单数据，参赛选手通过人工智能技术构建预测模型，预估用户人群在规定时间内产生购买行为的概率。

* 数据字段如下：

![](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/pkbigdata/master.other.img/c4b41b3a-9d2f-431e-9a31-d04bd8b9e338.png)
![](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/pkbigdata/master.other.img/e31fc70a-386f-4310-8716-a606b10ed23a.png)
![](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/pkbigdata/master.other.img/7bcfc1ae-03b6-46da-8c8d-b75cac6120c9.png)

* 评分标准

评分算法通过 **logarithmic loss**（记为 **logloss**）评估模型效果，**logloss** 越小越好。

![](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/pkbigdata/master.other.img/1890f2ec-229e-480c-9eb8-5c0b47cce7a6.png)

其中N表示测试集样本数量，$y_i$ 表示测试集中第i个样本的真实标签，$p_i$ 表示第 i个样本的预估转化率， δ为惩罚系数。

* 这里附上排名及分数，排名不是很高，特征工程做得不够多，需要多加学习。

![](https://upload-images.jianshu.io/upload_images/16911112-6b9b39163ac2ba35.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

