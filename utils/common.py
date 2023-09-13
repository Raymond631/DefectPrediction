import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_arff(folder_path, bug_label):
    """
    读取、合并arff数据集
    @param folder_path: 数据集路径
    @param bug_label: 数据集中标记为有bug的标签名
    @return: pandas数据帧
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
    combined_data.iloc[:, -1] = combined_data.iloc[:, -1].apply(lambda x: 1 if x == bug_label else 0)
    return combined_data

def read_arff_file(file_path,bug_label):
    data, meta = arff.loadarff(file_path)
    data_array = np.array(data.tolist())
    features = data_array[:, :-1]
    labels = data_array[:, -1]
    labels = np.where(labels == bug_label, 0, 1)
    return features, labels

# 文件夹下合并数据集处理
def csv_process(directory_path, bug_label1,bug_label2):
    csv_files=glob.glob(os.path.join(directory_path, '*.csv'))
    combined_data = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  # 请替换 'your_file.csv' 为你的文件路径
        # 将数据添加到合并的数据集中
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    labels = combined_data.iloc[:, -1].replace({bug_label1: 0, bug_label2: 1})
    features = combined_data.iloc[:, :-1]
    return features,labels
# 单个csv数据集文件处理
def csv_file_process(file_path, bug_label1,bug_label2):
    df = pd.read_csv(file_path)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1].replace({bug_label1: 0, bug_label2: 1})
    return features, labels

def data_split(df):
    """
    数据分割
    @param df: pandas数据帧
    @return: X_train, X_test, y_train, y_test
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def data_standard_scaler(X_train, X_test):
    """
    标准化特征数据
    @param X_train: 训练集-特征变量
    @param X_test: 测试集-特征变量
    @return:
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


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
    print(classification_report(y_test, y_pred, target_names=['clean', 'buggy']))

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
