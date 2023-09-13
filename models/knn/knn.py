# coding=utf-8
from __future__ import division

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from utils.common import read_arff, data_standard_scaler, data_split, model_evaluation


def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - data_set  # 扩充input_vct到与data_set同型并相减
    sq_diff_mat = diff_mat ** 2  # 矩阵中每个元素都平方
    distance = sq_diff_mat.sum(axis=1) ** 0.5  # 每行相加求和并开平方根
    return distance.min(axis=0)  # 返回最小距离


def file2mat(test_filename, para_num):
    """
    将表格存入矩阵，test_filename为表格路径，para_num为存入矩阵的列数
    返回目标矩阵，和矩阵每一行数据的类别
    """
    fr = open(test_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    result_mat = np.zeros((line_nums, para_num))  # 创建line_nums行，para_num列的矩阵

    class_label = []

    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')

        result_mat[i, :] = item_mat[0: para_num]

        if item_mat[-1] == 'N':
            class_label.append(1)  # 表格中最后一列正常1 异常2的分类存入class_label
        else:
            class_label.append(2)
    fr.close()
    return result_mat, class_label


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    print(data_set_size)
    roc_rate = np.zeros((2, data_set_size))
    for i in range(data_set_size):
        if data_set[2][i] == 0:
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(86):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 0:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal  # 阈值以上正常点/全体正常的点
        roc_rate[1][j] = abnormal1 / abnormal  # 阈值以上异常点/全体异常点
    return roc_rate


def test(X_train, X_test, y_test):
    # 创建一个1行2列的画布
    figure, axes = plt.subplots(ncols=2, nrows=1, figsize=(9.6, 4.5), dpi=100)
    # 绘图对象
    ax1 = axes[0]
    ax2 = axes[1]

    # training_mat, training_label = file2mat(training_filename, 32)
    # test_mat, test_label = file2mat(test_filename, 32)
    test_size = X_test.shape[0]  # shape[0]矩阵行数
    result = np.zeros((test_size, 3))
    for i in range(test_size):
        result[i] = i + 1, classify(X_test[i], X_train), y_test[i]  # 序号， 最小欧氏距离， 测试集数据类别
    result = np.transpose(result)  # 矩阵转置
    joblib.dump(result, "../../../files/dnn.pkl")
    # file = open('knn-result.txt', 'w')
    # file.write(str(test_label));
    # file.close()
    # np.savetxt('knn-result.txt', test_label)
    # 选择ax1
    plt.sca(ax1)
    plt.scatter(result[0], result[1], c=result[2], edgecolors='None', s=1, alpha=1)
    # 图1 散点图：横轴为序号，纵轴为最小欧氏距离，点中心颜色根据测试集数据类别而定， 点外围无颜色，点大小为最小1，灰度为最大1
    roc_rate = roc(result)

    # 选择ax2
    plt.sca(ax2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=4, alpha=1)
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR（真阳性率）')
    plt.xlabel('FPR（伪阳性率）')
    # 解决中文乱码和正负号问题
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # 图2 ROC曲线， 横轴误报率，即阈值以上正常点/全体正常的点；纵轴检测率，即阈值以上异常点/全体异常点
    plt.show()


if __name__ == "__main__":
    # test('../../../data/csv/MDP/D1/PC2.csv', '../../../data/csv/MDP/D1/PC5.csv')
    df = read_arff('../../../data/arff/SOFTLAB', b'buggy')
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)
    test(X_train, X_test, y_test)
    knn_model = joblib.load('../../../files/dnn.pkl')
    # 使用模型进行预测
    n_neighbors = 5
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]

    # 测试模型
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)
