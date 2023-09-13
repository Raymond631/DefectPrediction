from collections import Counter

import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from models.mlp.mlp import data_process


def test_adtree(file_path):
    x_test, labels = data_process(file_path)
    clf = joblib.load("../../files/dt.pkl")
    pre = clf.predict(x_test)
    # np.savetxt('network_result.txt', pre)
    print('预测结果', pre)
    print('准确率', accuracy_score(labels, pre))


def adtree_result(file_path):
    datasets, labels = data_process(file_path)
    clf = joblib.load("../../files/adtree.pkl")
    pre = clf.predict(datasets)
    Counter(pre)
    Yes = sum(pre == 1)
    No = sum(pre == 0)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(6, 6))
    label = ['有缺陷数', '无缺陷数']
    explode = [0.01, 0.05]
    values = [Yes, No]
    plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')  # 绘制饼图
    plt.title('缺陷数目')
    plt.show()


if __name__ == '__main__':
    file_path = '../../data/csv/MDP/D1/PC5.csv'  # 替换成你的目录路径
    test_adtree(file_path)
    adtree_result(file_path)
