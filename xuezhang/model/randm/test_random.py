# -*- coding: utf-8 -*-

from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

from xuezhang.model.randm.mdp_random import data_handle


# 缺陷预测
def test_random():
    datasets, labels = data_handle('MDP/KC4.csv')  # 对数据集进行处理
    X_test = datasets[:]
    clf0 = joblib.load("./files/random.pkl")
    y_predict = clf0.predict(X_test)  # 使用分类器对测试集进行预测
    np.savetxt('random_result.txt', y_predict)
    print(y_predict)


# 画饼状图
def random_result():
    datasets, labels = data_handle('MDP/KC4.csv')  # 对数据集进行处理
    X_test = datasets[:]
    clf0 = joblib.load("./files/random.pkl")
    y_predict = clf0.predict(X_test)  # 使用分类器对测试集进行预测

    Counter(y_predict)  # {label:sum(label)}
    Yes = sum(y_predict == 1)
    No = sum(y_predict == 0)
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆
    label = ['有缺陷数', '无缺陷数']  # 定义饼图的标签，标签是列表
    explode = [0.01, 0.05]  # 设定各项距离圆心n个半径
    values = [Yes, No]
    plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')  # 绘制饼图
    plt.title('缺陷数目')
    plt.show()


if __name__ == '__main__':
    test_random()
