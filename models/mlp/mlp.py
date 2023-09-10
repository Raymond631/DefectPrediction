import glob
import pandas as pd
import math
import os
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from utils.model.randm.mdp_random import data_handle

def plot_roc(labels, predict_prob, auca, preci, recall, f1, auc_ave, g_mean_ave, balance_ave):
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

def dataset_process(directory_path):
    csv_files=glob.glob(os.path.join(directory_path, '*.csv'))
    combined_data = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  # 请替换 'your_file.csv' 为你的文件路径
        # 将数据添加到合并的数据集中
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    labels = combined_data.iloc[:, -1]
    features = combined_data.iloc[:, :-1]
    modified_labels = [0 if label == "buggy" else 1 for label in labels]
    labels=pd.DataFrame(modified_labels,columns=['defects'])
    return features,labels


def multilayer_perceptron():
    # 指定目标目录的路径
    directory_path = '../../data/csv/AEEEM'  # 替换成你的目录路径
    features,labels = dataset_process(directory_path)
    print(type(features))
    print(type(labels))
    # 使用随机欠采样
    rus = RandomUnderSampler(sampling_strategy=1, random_state=0, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(features, labels)

    # k折交叉验证
    kf = StratifiedKFold(n_splits=10, shuffle=True)

    # 定义mlp的分类器
    clf = MLPClassifier(hidden_layer_sizes=(10,1), activation='tanh', solver='sgd', alpha=0.001,
                        batch_size=5, learning_rate='constant', learning_rate_init=0.03, power_t=0.5, max_iter=200,
                        shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

    # 评估指标列表
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    g_mean_list = []
    balance_list = []
    for train_index, test_index in kf.split(X_resampled, y_resampled):
        # 使用 train_index 来获取训练集的索引
        x_train, x_val = features.iloc[train_index], features.iloc[test_index]
        y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]
        clf.fit(x_train, y_train)
        joblib.dump(clf, "../../files/mlp.pkl")
        # 使用验证集预测结果
        pre = clf.predict(x_val)
        y_score=clf.predict_proba(x_val)
        y_score = y_score[:, 1]
        # 计算评估指标
        accuracy = accuracy_score(y_val, pre)
        precision = precision_score(y_val, pre, average='weighted')
        recall = recall_score(y_val, pre, average='weighted')
        f1score = f1_score(y_val, pre, average='weighted')
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        false_positive_rate, true_positive_rate = false_positive_rate[1], true_positive_rate[1]
        # 几何平均
        g_mean = math.sqrt(true_positive_rate * (1 - false_positive_rate))
        # 平衡度（Balance）是一种用于评估分类模型的平衡性的指标。它的计算方式是根据真阳性率和伪阳性率的欧几里得距离来衡量，除以根号 2。
        balance = 1 - math.sqrt(math.pow((1 - true_positive_rate), 2) + math.pow((0 - false_positive_rate), 2)) / math.sqrt(2)
        print('准确率：', accuracy_score(y_val, pre))
        print('分类报告：', classification_report(y_val, pre))
        # 将评估指标添加到列表中
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1score)
        auc_list.append(roc_auc)
        g_mean_list.append(g_mean)
        balance_list.append(balance)


    # 计算平均评估指标
    auca = np.mean(accuracy_list)
    preci = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
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
    print('混淆矩阵输出:\n', metrics.confusion_matrix(y_val, pre))  # 混淆矩阵输出
    # y_prob=clf.predict_proba(X_resampled)
    # y_prob = y_prob[:, 1]
    plot_roc(y_val, y_score, auca, preci, recall, f1, auc_ave, g_mean_ave, balance_ave)  # 绘制ROC曲线并求出AUC值

if __name__ == '__main__':
    multilayer_perceptron()