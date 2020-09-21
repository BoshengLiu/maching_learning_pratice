import time
from data_process import *
from model_func import *

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv', low_memory=False)

    start = time.time()
    # 分析数据
    # print(data.isna().sum())
    # dataView(data)

    # 数据清洗
    data.drop(columns=['customer_province', 'customer_city'], inplace=True)
    data['order_detail_status'].replace(501, 5, inplace=True)
    data['order_detail_status'].replace(101, 5, inplace=True)
    data['order_status'].replace(101, 5, inplace=True)

    # 获取消费者所有id
    customers = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()

    # 将时间格式转换为datetime
    data['order_pay_time'] = pd.to_datetime(data['order_pay_time'])

    # 获取新的列-日期
    data['order_pay_date'] = data['order_pay_time'].dt.date

    # 划分训练集和测试集
    off_train = data[data['order_pay_date'].astype(str) <= '2013-06-31']
    off_test = data[(data['order_pay_date'].astype(str) > '2013-06-31') & (data['order_pay_date'].values != 'null')]
    online_train = data

    # 数据处理
    train_df = dataProcess(off_train, off_test, False)
    test_df = dataProcess(online_train, None, True)
    y = train_df.pop('label')
    features = [x for x in train_df.columns if x not in ['customer_id']]
    x = train_df[features]
    test = test_df[['customer_id']]
    x_test = test_df[features]

    # 训练模型
    # rfcTraining(x, y)

    # 预测模型
    test['result'] = xgbPredict(x, y, x_test)

    # 写入数据
    test = test.drop(0)
    test.to_csv('xgb_sample.csv', index=False)

    end = time.time()
    using_time = end - start
    print("Process finish.Using %d min %.4f second!" % (using_time // 60, using_time % 60))
