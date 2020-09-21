from feature_selector import FeatureSelector
from modelFunc import *
from dataPreprocess import *

if __name__ == '__main__':
    start = time.time()

    result = []
    power_list = [20, 30, 10, 20, 21, 10, 40, 30, 50, 20]  # 电场装机功率
    for num, p in zip(range(1, 11), power_list):
        print("第%s个电场..." % num)
        # 训练集数据整理
        train = pd.read_csv("train/train_%s.csv" % num)

        # 官方要求数据清洗
        train = train[train["实际辐照度"] >= 0].drop("实际辐照度", axis=1)
        train["时间"] = pd.to_datetime(train["时间"], format='%Y/%m/%d %H:%M')
        if num == 7:
            train = train[(train["时间"] < "2018/03/01 00:00") | (train["时间"] > "2018/04/04 23:45")]
        if num == 9:
            train = train[(train["时间"] < "2016/01/01 9:00") | (train["时间"] > "2017/03/21 23:45")]
        train["15分钟段"] = train["时间"].dt.time

        # 清洗前数据可视化
        saveBeforeclean(train, num)

        # 数据初步处理+数据清洗
        train = processData(train)
        limit_down, limit_up = np.percentile(train['实际功率'], [5, 95])
        train = train[(train['实际功率'] < limit_up) & (train['实际功率'] > limit_down)].reset_index(drop=True)

        # 清洗后数据可视化
        saveAfterclean(train, num)

        # 处理测试集数据
        test = pd.read_csv("test/test_%s.csv" % num)
        test["时间"] = pd.to_datetime(test["时间"], format='%Y/%m/%d %H:%M')
        test["15分钟段"] = test["时间"].dt.time
        test = processData(test)

        # 处理天气数据
        weather = pd.read_csv("气象数据/电站%s_气象.csv" % num, encoding="gbk")
        weather["时间"] = pd.to_datetime(weather["时间"], format='%Y-%m-%d %H:%M')

        # 合并所有数据
        data = pd.concat([train, test], ignore_index=True, sort=False)
        data.to_csv('train_base/train_%s.csv' % num, index=False)

        # 转换15分钟段数据
        data = data.merge(weather[["时间", "直辐射"]], left_on=["时间"], right_on=["时间"], how="left")
        period = data.groupby("15分钟段", as_index=False)["实际功率"].agg({"base_mean": "mean"}).reset_index()
        period["15分钟段_num"] = np.abs(period["index"] - period["base_mean"].values.argmax())
        period = period.fillna(0)
        data = data.merge(period[["15分钟段", "15分钟段_num", "base_mean"]], left_on="15分钟段", right_on="15分钟段")

        # 构造新变量
        vars = ['时角', '高度角', '辐照度', '风速', '风向', '温度', '湿度', '压强', 'base_mean', '15分钟段_num', '直辐射']
        for var1 in vars:
            for var2 in ['辐照度', 'base_mean', '15分钟段_num', '直辐射', '温度', '湿度']:
                data['%s_%s' % (var1, var2)] = np.multiply(data[var1], data[var2])

        not_x = ['15分钟段', 'id', '时间', '实际功率']
        x_train = data[np.isnan(data['id'])].drop(not_x, axis=1)
        y_train = data[np.isnan(data['id'])]['实际功率']
        x_test = data[data['id'] > 0].drop(['15分钟段', 'id', '时间', '实际功率'], axis=1)

        # 特征选择-方法1
        fs = FeatureSelector(data=x_train, labels=y_train)
        fs.identify_collinear(correlation_threshold=0.99)
        choose = fs.ops['collinear']
        x_train_select = x_train.drop(choose, axis=1)
        x_test_select = x_test.drop(choose, axis=1)

        # 特征选择-方法2
        # fs = featureChoice(x_train, y_train, num)
        # x_train_select = x_train[fs]
        # x_test_select = x_test[fs]

        # 将测试集中不存在的列而训练集中存在的列删除
        for var1 in ['辐照度', 'base_mean', '15分钟段_num', '直辐射', '']:
            for var2 in x_test_select.columns:
                if var1 not in var2:
                    x_train = x_train_select.drop(var2, axis=1)
                    x_test = x_test_select.drop(var2, axis=1)

        # 训练并预测-lgb
        # params = lgbTraining(x_train, y_train, p)[1]
        # y_predict = lgbPredict(x_train, y_train, x_test, lgb_params[num - 1])

        # 训练并预测-stackModel，10个模型，需要向垂直方向堆叠成新的数组，然后选择10个中的中值为预测的结果
        y_predict = stackModel(x_train, y_train, x_test)
        y_predict = np.vstack(y_predict)
        y_predict = np.median(y_predict, axis=0)

        # 预测结果可视化
        predictShow(y_predict, num)

        # 将预测结果生成DataFrame并保存
        temp_predict = pd.DataFrame({"id": data[data['id'] > 0]['id'].values, "prediction": y_predict})
        result.append(temp_predict)
        temp_predict.to_csv("predict/base_%s.csv" % num, index=False)

    # 将10个电场的预测结果拼接并去重，保存为csv文件
    result = pd.concat(result, ignore_index=True)
    result['id'] = result['id'].astype(np.int)
    result = result.sort_values(by="id").drop_duplicates()
    result.to_csv("sample.csv", index=False)

    end = time.time()
    time_spending = end - start
    print('Process finished! Using %d min and %.4f sec.' % (time_spending // 60, time_spending % 60))
