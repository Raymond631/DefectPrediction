# 模型：LR
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation


def train_lr(X_train, Y_train):
    # 创建LR分类器
    lr_model = LogisticRegression()
    # 训练模型
    lr_model.fit(X_train, Y_train)  # 调用LogisticRegression中的fit函数训练模型参数
    joblib.dump(lr_model, '../../files/lr.pkl')
def test_lr(X_test):
    # 加载模型
    lr_model = joblib.load('../../files/lr.pkl')
    # 使用模型进行预测
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    return lr_pred, lr_prob


def lr(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_lr(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_lr(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    lr('../../data/arff/AEEEM', b'buggy')
