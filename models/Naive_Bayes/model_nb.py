import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, roc_curve, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from xuezhang.model.randm.mdp_random import data_handle
import matplotlib as mpl
import os
import glob

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



if __name__ == '__main__':
    # data = arff.loadarff('../../data/arff/AEEEM/EQ.arff')
    # df = pd.DataFrame(data[0])
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    # 输入数据的修改
    directory_path = '../../data/csv/AEEEM'
    # 使用 glob.glob 获取目录下所有 CSV 文件的路径列表
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    # 打印所有 CSV 文件的路径
    for file in csv_files:
        print(file)
    X, y = data_handle('../../data/csv/AEEEM/LC.csv')

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

    print('分类报告：', classification_report(y_test, y_pred))
    plot_roc(y_test, y_pred, auc, macro, macro_recall, weighted)   # 绘制ROC曲线并求出AUC值
    print('结果')
    print(y_test)
    print(y_pred)
