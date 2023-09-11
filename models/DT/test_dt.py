# 单个arff数据集文件处理
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import accuracy_score

from models.mlp.mlp import data_process


def read_arff_file(file_path):
    data, meta = arff.loadarff(file_path)
    data_array = np.array(data.tolist())
    features = data_array[:, :-1]
    labels = data_array[:, -1]
    labels = np.where(labels == b'N', 0, 1)
    return features, labels


def folder_arff(folder_path):
    arff_files = [f for f in os.listdir(folder_path) if f.endswith('.arff')]
    combined_data = pd.DataFrame()
    for filename in arff_files:
        file_path = os.path.join(folder_path, filename)
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    combined_data.iloc[:, -1] = combined_data.iloc[:, -1].apply(lambda x: 0 if x == b'N' else 1)
    X = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1].astype(int)
    return X, y
def test_dt(file_path):
    x_test, labels = data_process(file_path)
    clf = joblib.load("../../files/dt.pkl")
    pre = clf.predict(x_test)
    #np.savetxt('network_result.txt', pre)
    print('预测结果',pre)
    print('准确率',accuracy_score(labels, pre))

def dt_result(file_path):
    datasets, labels = data_process(file_path)
    clf = joblib.load("../../files/dt.pkl")
    pre = clf.predict(datasets)
    Counter(pre)
    Yes = sum(pre == 1)
    No = sum(pre == 0)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(6, 6))
    label = ['有缺陷数', '无缺陷数']
    explode = [0.01, 0.05]
    values = [Yes, No]
    plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')  # 绘制饼图
    plt.title('缺陷数目')
    plt.show()

if __name__ == '__main__':
    file_path = '../../data/csv/MDP/D1/PC5.csv'  # 替换成你的目录路径
    test_dt(file_path)
    dt_result(file_path)