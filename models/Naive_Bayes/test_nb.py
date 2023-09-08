import matplotlib.pyplot as plt
import numpy as np
from model_nb import NaiveBayes
from scipy.io import arff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pandas as pd

if __name__ == '__main__':
    data = arff.loadarff('../../data/AEEEM/EQ.arff')
    df = pd.DataFrame(data[0])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(y)
    for i in range(len(y)):
        if y[i] == 'b\'clean\'':
            y[i] = 1
        else:
            y[i] = 0

    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    print('数据集测试')
    print(X)
    print(y)
    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # 创建并训练分类器
    clf = GaussianNB(var_smoothing=1e-9)
    # classifier = NaiveBayes()
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    acc = np.sum(y_test == y_pred) / X_test.shape[0]
    print("Test Acc : %.3f" % acc)

    # 预测
    y_proba = clf.predict_proba(X_test[:1])
    print(clf.predict(X_test[:1]))
    print("预计的概率值:", y_proba)

    # 创建一个等差数列作为横坐标
    # x = np.arange(len(y_new))
    #
    # # 绘制实际结果和预测结果的曲线
    # plt.plot(x, y_new, label='实际结果')
    # plt.plot(x, y_pred, label='预测结果')
    #
    # # 添加图例和标签
    # plt.legend()
    # plt.xlabel('样本')
    # plt.ylabel('类别')
    #
    # # 显示图形
    # plt.show()