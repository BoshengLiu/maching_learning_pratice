import numpy as np
import pandas as pd
from feature_selector import FeatureSelector


# 查看每列不同的属性值并保存
def dataView(df):
    for i in df.columns:
        path = 'label_values/' + str(i) + '.csv'
        count_classes = pd.value_counts(df[i])

        index = pd.DataFrame(count_classes.index, columns=[i])
        values = pd.DataFrame(count_classes.values, columns=['num'])
        dt = pd.concat([index, values], axis=1)

        dt.to_csv(path, index=False)
        print('File saved!')


# 数据预处理
def dataProcess(train, test, label):
    # 用户购买同种货物的次数goods_num
    goods_num = train[['customer_id', 'goods_id', 'order_pay_date', 'order_detail_goods_num']]
    goods_num = goods_num[goods_num['order_pay_date'].values != 'null']
    goods_num = goods_num.groupby(['customer_id', 'goods_id'])['order_detail_goods_num'].agg('sum').reset_index()
    goods_num = goods_num.groupby('customer_id')['goods_id', 'order_detail_goods_num'].agg('max')
    goods_num.rename(columns={'goods_id': 'customer_max_goods'}, inplace=True)

    # 用户下单数order_num
    order_num = train[['customer_id', 'order_id', 'order_pay_date']]
    order_num = order_num[order_num['customer_id'] != 'null']
    order_num['user_order_num'] = 1
    order_num = order_num.groupby('customer_id').agg('sum').reset_index()

    # 商品价格goods_price
    goods_price = train[['customer_id', 'goods_price', 'order_pay_date']]
    goods_price = goods_price[goods_price.order_pay_date != 'null']
    goods_price = goods_price.groupby('customer_id')['goods_price'].agg(
        {'price_mean': 'mean', 'price_max': 'max', 'price_min': 'min'})

    # 用户的订单购买时间order_time
    order_time = train[['customer_id', 'order_pay_date']]
    order_time = order_time[order_time['order_pay_date'] != 'null']
    order_time = order_time.groupby(['customer_id'], as_index=False)['order_pay_date'].agg(
        {'order_pay_date_first': 'min', 'order_pay_date_last': 'max'})
    order_time['order_time'] = pd.to_datetime(order_time['order_pay_date_last']) - pd.to_datetime(
        order_time['order_pay_date_first'])
    order_time['order_time'] = order_time['order_time'].dt.days + 1

    # 用户是否喜欢参与评价，将空值填充为0
    user_evaluate = train[['customer_id', 'is_customer_rate']]
    user_evaluate = user_evaluate.replace(np.nan, 0)
    user_evaluate = user_evaluate.groupby('customer_id')['is_customer_rate'].agg('sum').reset_index()

    # 父订单商品购买数量
    father_order_num = train[['customer_id', 'order_total_num', 'order_pay_date']]
    father_order_num = father_order_num[father_order_num['order_pay_date'] != 'null']
    father_order_num = father_order_num.groupby('customer_id')['order_total_num'].agg({'order_num_mean': 'mean'})

    # 父订单商品实际付款数量
    father_order_buy = train[['customer_id', 'order_total_payment']]
    father_order_buy = father_order_buy.groupby('customer_id')['order_total_payment'].agg({'order_pay_mean': 'mean'})

    # 父订单优惠金额
    discount_father = train[['customer_id', 'order_total_discount']]
    discount_father = discount_father.groupby('customer_id')['order_total_discount'].agg(
        {'order_discount_mean': 'mean'})

    # 子订单商品购买数量，将空值填充为0
    son_order_buy = train[['customer_id', 'order_detail_amount']]
    son_order_buy = son_order_buy.groupby('customer_id')['order_detail_amount'].agg({'order_amount_mean': 'mean'})
    son_order_buy = son_order_buy.fillna(0)

    # 子订单应付总金额
    son_order_price = train[['customer_id', 'order_detail_payment']]
    son_order_price = son_order_price.groupby('customer_id')['order_detail_payment'].agg({'detail_pay_mean': 'mean'})

    # 是否支持会员折扣
    vip_discount = train[['customer_id', 'goods_has_discount']]
    vip_discount = vip_discount.groupby('customer_id')['goods_has_discount'].agg({'goods_discount_mean': 'mean'})

    # 用户状态，将空值填充为0
    customer_status = train[['customer_id', 'customer_gender', 'member_status']].drop_duplicates(['customer_id'])
    customer_status = customer_status.fillna(0)

    # 父订单子订单数
    son_order_num = train[['customer_id', 'order_count']]
    son_order_num = son_order_num.groupby('customer_id')['order_count'].agg({'order_count_mean': 'mean'})

    # 父订单状态
    father_order_status = train[['customer_id', 'order_status']]
    father_order_status = father_order_status.groupby('customer_id')['order_status'].agg({'order_status_mean': 'mean'})

    # 特征组合
    features_set = pd.merge(goods_num, order_num, on='customer_id', how='left')
    features_set = pd.merge(features_set, goods_price, on='customer_id', how='left')
    features_set = pd.merge(features_set, order_time, on='customer_id', how='left')
    features_set = pd.merge(features_set, user_evaluate, on='customer_id', how='left')
    features_set = pd.merge(features_set, father_order_num, on='customer_id', how='left')
    features_set = pd.merge(features_set, father_order_buy, on='customer_id', how='left')
    features_set = pd.merge(features_set, discount_father, on='customer_id', how='left')
    features_set = pd.merge(features_set, son_order_buy, on='customer_id', how='left')
    features_set = pd.merge(features_set, son_order_price, on='customer_id', how='left')
    features_set = pd.merge(features_set, vip_discount, on='customer_id', how='left')
    features_set = pd.merge(features_set, customer_status, on='customer_id', how='left')
    features_set = pd.merge(features_set, father_order_status, on='customer_id', how='left')
    features_set = pd.merge(features_set, son_order_num, on='customer_id', how='left')
    features_set.drop_duplicates(['customer_id'])

    # 删除第一次下单时间
    del features_set['order_pay_date_first']

    # 构建标签
    if label == False:
        features_set['order_pay_date_last'] = pd.to_datetime(test['order_pay_date'].min()) - pd.to_datetime(
            features_set['order_pay_date_last'])
        features_set['order_pay_date_last'] = features_set['order_pay_date_last'].dt.days + 1
        features_set['label'] = 0
        features_set.loc[features_set['customer_id'].isin(list(test['customer_id'].unique())), 'label'] = 1
    else:
        features_set['order_pay_date_last'] = pd.to_datetime('2013-12-31') - pd.to_datetime(
            features_set['order_pay_date_last'])
        features_set['order_pay_date_last'] = features_set['order_pay_date_last'].dt.days + 1
    return features_set


def featureSelect(x, y):
    fs = FeatureSelector(data=x, labels=y)
    fs.identify_collinear(correlation_threshold=0.99)
    choose = fs.ops['collinear']
    return choose
