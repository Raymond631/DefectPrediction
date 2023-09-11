import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score


def model_evaluation(y_test, y_pred, y_prob):
    """
    模型评估
    @param y_test: 测试集-目标变量
    @param y_pred: 预测结果
    @param y_prob: 预测概率
    """
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # 分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['buggy', 'clean']))

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def read_arff(folder_path, bug_label):
    """
    读取、合并arff数据集
    @param folder_path: 数据集路径
    @param bug_label: 数据集中标记为有bug的标签名
    @return: pandas data frame
    """
    # 获取目录下的所有ARFF文件
    arff_files = [f for f in os.listdir(folder_path) if f.endswith('.arff')]
    combined_data = pd.DataFrame()
    # 读取数据
    for filename in arff_files:
        file_path = os.path.join(folder_path, filename)
        # 从ARFF文件加载数据
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        # 将数据添加到合并的数据集中
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    # 将标签列转换为二进制（0和1）
    combined_data.iloc[:, -1] = combined_data.iloc[:, -1].apply(lambda x: 0 if x == bug_label else 1)
    return combined_data
