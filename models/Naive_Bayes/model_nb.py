import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn import metrics
from sklearn.metrics import auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


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


def naive_Bayes(folder_path):
    # 获取目录下的所有ARFF文件
    arff_files = [f for f in os.listdir(folder_path) if f.endswith('.arff')]
    combined_data = pd.DataFrame()

    for filename in arff_files:
        file_path = os.path.join(folder_path, filename)
        # 从ARFF文件加载数据
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        # 将数据添加到合并的数据集中
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    # 使用LabelEncoder将字符串目标变量转换为数值
    label_encoder = LabelEncoder()
    combined_data['class'] = label_encoder.fit_transform(combined_data['class'])
    class_labels = [label.decode('utf-8') for label in label_encoder.classes_]  # 保存原始的枚举类型标签

    # 分割数据为特征 (X) 和目标变量 (y)
    X = combined_data.iloc[:, :-1]
    y = combined_data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练分类器
    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    auc = metrics.accuracy_score(y_test, y_pred)
    macro = metrics.precision_score(y_test, y_pred, average='macro')
    micro = metrics.precision_score(y_test, y_pred, average='micro')
    macro_recall = metrics.recall_score(y_test, y_pred, average='macro')
    weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    # 预测
    y_proba = clf.predict_proba(X_test[:1])
    print(clf.predict(X_test[:1]))
    print("预计的概率值:", y_proba)

    print('分类报告：', classification_report(y_test, y_pred, target_names=class_labels))
    plot_roc(y_test, y_pred, auc, macro, macro_recall, weighted)  # 绘制ROC曲线并求出AUC值
    print('结果')
    print(y_test)
    print(y_pred)


if __name__ == '__main__':
    naive_Bayes('../../data/arff/AEEEM')
