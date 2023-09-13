# -*- coding: utf-8 -*-

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

"""
函数说明：文件处理
Parameters:
     filename:数据文件
Returns:
     list_datasets：数据集特征列表
     category_labels:数据标签列表
"""


def data_handle(filename):
    read_data = pd.read_csv(filename)
    list_datasets = []
    category_labels = []
    for i in range(len(read_data)):
        list_data = []
        for j in range(len(read_data.iloc[i, :]) - 1):
            row_data = read_data.iloc[i, j]  # 读取每个样本的每个数据
            list_data.append(row_data)  # 将每个数据存入列表
        list_datasets.append(list_data)  # 将每个样本的数据存入列表

        row_data_label = read_data.iloc[i, len(read_data.iloc[i, :]) - 1]  # 读取每个样本的类别标签
        if row_data_label == 'N':
            category_labels.append(0)  # 将二分类标签转化为0和1,0代表软件正常，1代表软件缺陷
        else:
            category_labels.append(1)
    return list_datasets, category_labels


"""
函数说明：绘制ROC曲线
Parameters:
     labels:测试标签列表
     predict_prob:预测标签列表
"""


def plot_roc(labels, predict_prob, auc, macro, macro_recall, weighted):
    # 创建一个1行2列的画布
    figure, axes = plt.subplots(ncols=1, nrows=2, figsize=(6.5, 6.5), dpi=100)
    # 绘图对象
    ax1 = axes[0]
    ax2 = axes[1]

    # 选择ax1
    plt.sca(ax1)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, predict_prob)  # 真阳性，假阳性，阈值
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)  # 计算AUC值
    print('AUC=' + str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR（真阳性率）')
    plt.xlabel('FPR（伪阳性率）')

    # 选择ax2
    plt.sca(ax2)
    plt.axis('off')
    plt.title('模型评价指标', y=-0.1)
    # 解决中文乱码和正负号问题
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    col_labels = ['准确率', '精确率', '召回率', 'f1值']
    row_labels = ['期望', '实际']
    table_vals = [[0.9, 0.8, 0.75, 0.8], [auc, macro, macro_recall, weighted]]
    row_colors = ['red', 'pink', 'green', 'gold']
    table = plt.table(cellText=table_vals, colWidths=[0.18 for x in col_labels],
                      rowLabels=row_labels, colLabels=col_labels,
                      rowColours=row_colors, colColours=row_colors,
                      loc="center")
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    plt.show()
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存


def random_forest():
    datasets, labels = data_handle('../../../data/csv/MDP/D1/PC1.csv')  # 对数据集进行处理

    # 训练集和测试集划分
    X_train = datasets[:115]  # 第0到115个数
    y_train = labels[:115]
    X_test = datasets[90:]  # 第90到最后一个数
    y_test = labels[90:]

    # 随机森林分类器
    clf = RandomForestClassifier()
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)  # 使用训练集对分类器训练
    joblib.dump(clf, "../../../files/random.pkl")
    y_predict = clf.predict(X_test)  # 使用分类器对测试集进行预测

    auc = metrics.accuracy_score(y_test, y_predict)
    macro = metrics.precision_score(y_test, y_predict, average='macro')
    micro = metrics.precision_score(y_test, y_predict, average='micro')
    macro_recall = metrics.recall_score(y_test, y_predict, average='macro')
    weighted = metrics.f1_score(y_test, y_predict, average='weighted')
    print('准确率:', auc)  # 预测准确率输出
    print('宏平均精确率:', macro)  # 预测宏平均精确率输出
    print('微平均精确率:', micro)  # 预测微平均精确率输出
    print('宏平均召回率:', macro_recall)  # 预测宏平均召回率输出
    print('平均F1-score:', weighted)  # 预测平均f1-score输出
    print('混淆矩阵输出:\n', metrics.confusion_matrix(y_test, y_predict))  # 混淆矩阵输出
    print('分类报告:', metrics.classification_report(y_test, y_predict))  # 分类报告输出
    plot_roc(y_test, y_predict, auc, macro, macro_recall, weighted)  # 绘制ROC曲线并求出AUC值


if __name__ == '__main__':
    random_forest()
