import glob
import os
import tkinter as tk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import arff
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
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


def read_arff_file(file_path, bug_label):
    data, meta = arff.loadarff(file_path)
    data_array = np.array(data.tolist())
    features = data_array[:, :-1]
    labels = data_array[:, -1]
    labels = np.where(labels == bug_label, 0, 1)
    return features, labels


# 文件夹下合并数据集处理
def csv_process(directory_path, bug_label1, bug_label2):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    combined_data = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  # 请替换 'your_file.csv' 为你的文件路径
        # 将数据添加到合并的数据集中
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    labels = combined_data.iloc[:, -1].replace({bug_label1: 0, bug_label2: 1})
    features = combined_data.iloc[:, :-1]
    return features, labels


# 单个csv数据集文件处理
def csv_file_process(file_path, bug_label1, bug_label2):
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
    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=['clean', 'buggy'])

    # 生成ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 生成预测结果饼状图
    labels = ['clean', 'buggy']
    sizes = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
    explode = (0.1, 0)  # 突出第一个分片
    colors = ['#ff9999', '#66b3ff']  # 饼状图颜色
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # 保证饼状图是正圆形

    # 创建Tkinter窗口
    root = tk.Tk()
    root.title("Matplotlib in Tkinter")
    # 设置窗口大小
    root.geometry("1200x800")
    # 创建一个框架以包含Matplotlib图
    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # 创建一个Matplotlib图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 在子图1中显示分类报告
    axs[0, 0].text(0.1, 0.5, report, fontsize=12)
    axs[0, 0].axis('off')

    # 在子图2中显示ROC曲线
    axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0, 1].set_xlim([0.0, 1.0])
    axs[0, 1].set_ylim([0.0, 1.05])
    axs[0, 1].set_xlabel('False Positive Rate')
    axs[0, 1].set_ylabel('True Positive Rate')
    axs[0, 1].set_title('Receiver Operating Characteristic')
    axs[0, 1].legend(loc='lower right')

    # 在子图3中显示混淆矩阵
    axs[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[1, 0].set_title('Confusion Matrix')
    axs[1, 0].set_xlabel('Predicted')
    axs[1, 0].set_ylabel('True')
    tick_marks = np.arange(len(labels))
    axs[1, 0].set_xticks(tick_marks)
    axs[1, 0].set_yticks(tick_marks)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axs[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    # 在子图4中显示预测结果饼状图
    axs[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    axs[1, 1].axis('equal')

    # 创建一个Canvas，用于在Tkinter窗口中显示Matplotlib图
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # 运行Tkinter主循环
    root.mainloop()
