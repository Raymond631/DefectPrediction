# -*- coding: utf-8 -*-
import math

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold  # 分层k折交叉验证
from sklearn.neural_network import MLPClassifier

from xuezhang.model.randm.mdp_random import data_handle


def plot_roc(labels, predict_prob, auca, preci, recall, f1, auc_ave, g_mean_ave, balance_ave):
    # 创建一个1行2列的画布
    figure, axes = plt.subplots(ncols=1, nrows=3, figsize=(7.5, 8), dpi=100)
    # 绘图对象
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # 选择ax1
    plt.sca(ax1)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    # print('AUC='+str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR（真阳性率）')
    plt.xlabel('FPR（伪阳性率）')
    # plt.savefig('figures/PC5.png')

    # 选择ax2
    plt.sca(ax2)
    plt.axis('off')
    plt.title('模型评价指标', y=-0.1)
    # 解决中文乱码和正负号问题
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    col_labels = ['准确率', '精确率', '召回率', 'f1值']
    row_labels = ['期望', '实际']
    table_vals = [[0.9, 0.8, 0.75, 0.8], [auca, preci, recall, f1]]
    row_colors = ['red', 'pink', 'green', 'gold']
    table = plt.table(cellText=table_vals, colWidths=[0.18 for x in col_labels],
                      rowLabels=row_labels, colLabels=col_labels,
                      rowColours=row_colors, colColours=row_colors,
                      loc="center")
    table.set_fontsize(14)
    table.scale(1.5, 1.5)

    # 选择ax3
    plt.sca(ax3)
    plt.axis('off')
    plt.title('收敛指标', y=-0.1)
    # 解决中文乱码和正负号问题
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    col_labels = ['Auc-Ave', 'G-Mean', 'Balance-Ave']
    row_labels = ['期望', '实际']
    table_vals = [[0.75, 0.7, 0.7], [auc_ave, g_mean_ave, balance_ave]]
    row_colors = ['yellow', 'cyan', 'green']
    table = plt.table(cellText=table_vals, colWidths=[0.22 for x in col_labels],
                      rowLabels=row_labels, colLabels=col_labels,
                      rowColours=row_colors, colColours=row_colors,
                      loc="center")
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    plt.show()


def nerual_network():
    datasets, labels = data_handle('MDP/KC3.csv')  # 对数据集进行处理
    '''
    x_train,x_test,y_train,y_test = train_test_split(datasets,labels,test_size=0.1,random_state=0) #数据集划分
    print(len(x_train))
    print(len(x_test))
    '''
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    g_mean_list = []
    balance_list = []

    # kf=KFold(n_splits=10)
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(datasets[:], labels[:]):
        rus = RandomUnderSampler(sampling_strategy=1, random_state=0, replacement=True)  # 采用随机欠采样（下采样）
        x_retest, y_retest = rus.fit_sample(datasets[:], labels[:])
        x_train = x_retest
        y_train = y_retest
        x_test = np.array(datasets)[test_index]
        y_test = np.array(labels)[test_index]

        clf = MLPClassifier(hidden_layer_sizes=(200), activation='tanh', solver='sgd', alpha=0.001,
                            batch_size=5, learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200,
                            shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                            beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        clf.fit(x_train, y_train)
        joblib.dump(clf, "./files/dnn.pkl")
        '''
        print(clf.n_iter_)
        print(clf.n_layers_)
        print(clf.n_outputs_)
        print(clf.batch_size)
        print(clf.loss)
        print(clf.activation)
        '''
        pre = clf.predict(x_test)
        print('准确率：', accuracy_score(y_test, pre))
        print('分类报告：', classification_report(y_test, pre))
        accuracy = accuracy_score(y_test, pre)
        precision = precision_score(y_test, pre, average='weighted')
        recall = recall_score(y_test, pre, average='weighted')
        f1score = f1_score(y_test, pre, average='weighted')
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pre)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        false_positive_rate, true_positive_rate = false_positive_rate[1], true_positive_rate[1]
        g_mean = math.sqrt(true_positive_rate * (1 - false_positive_rate))
        balance = 1 - math.sqrt(
            math.pow((1 - true_positive_rate), 2) + math.pow((0 - false_positive_rate), 2)) / math.sqrt(2)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1score)
        auc_list.append(roc_auc)
        g_mean_list.append(g_mean)
        balance_list.append(balance)

    # print('k-accuracy:', accuracy_list) #准确率
    # print('k-precision:', precision_list) #精确率
    # print('k-recall:', recall_list) #召回率
    # print('k-f1_score', f1_list) #F-score相当于precision和recall的调和平均
    # print('k-auc', auc_list)
    # print('k-g_mean:', g_mean_list)
    # print('k-balance:', balance_list)
    auca = np.mean(accuracy_list)
    preci = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    # 收敛标准
    auc_ave = np.mean(auc_list)
    g_mean_ave = np.mean(g_mean_list)
    balance_ave = np.mean(balance_list)
    print('平均准确率:', np.mean(accuracy_list))  # 准确率
    print('平均精确率:', np.mean(precision_list))  # 精确率
    print('平均召回率:', np.mean(recall_list))  # 召回率
    print('平均f1值:', np.mean(f1_list))
    # 收敛标准，一般大于0.7时采纳模型
    print('auc_ave:', np.mean(auc_list))
    print('g_mean_ave:', np.mean(g_mean_list))
    print('balance_ave:', np.mean(balance_list))

    print('混淆矩阵输出:\n', metrics.confusion_matrix(y_test, pre))  # 混淆矩阵输出

    plot_roc(y_test, pre, auca, preci, recall, f1, auc_ave, g_mean_ave, balance_ave)  # 绘制ROC曲线并求出AUC值


if __name__ == '__main__':
    nerual_network()
