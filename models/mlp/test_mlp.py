from collections import Counter
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.mlp.mlp import data_process


def test_mlp(file_path):
    x_test, labels = data_process(file_path)
    clf = joblib.load("../../files/mlp.pkl")
    pre = clf.predict(x_test)
    np.savetxt('network_result.txt', pre)
    print('预测结果',pre)

def dnn_result():
    datasets, labels = data_handle('MDP/KC3.csv')  # 对数据集进行处理

    # kf=KFold(n_splits=10)
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(datasets[:], labels[:]):
        x_test = np.array(datasets)[test_index]
        clf = joblib.load("../../../files/dnn.pkl")
    pre = clf.predict(x_test)
    Counter(pre)  # {label:sum(label)}
    Yes = sum(pre == 1)
    No = sum(pre == 0)
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.figure(figsize=(6, 6))  # 将画布设定为正方形，则绘制的饼图是正圆
    label = ['有缺陷数', '无缺陷数']  # 定义饼图的标签，标签是列表
    explode = [0.01, 0.05]  # 设定各项距离圆心n个半径
    values = [Yes, No]
    plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')  # 绘制饼图
    plt.title('缺陷数目')
    plt.show()

if __name__ == '__main__':
    file_path = '../../data/csv/AEEEM/EQ.csv'
    test_mlp(file_path)