# 引入必要的庫
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from models.mlp.mlp import dataset_process


def decision_tree():
    # 指定目标目录的路径
    directory_path = '../../data/csv/AEEEM'  # 替换成你的目录路径
    features,labels = dataset_process(directory_path)

    # 使用随机欠采样
    rus = RandomUnderSampler(sampling_strategy=1, random_state=0, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(features, labels)

    # k折交叉验证
    kf = StratifiedKFold(n_splits=10, shuffle=True)


    clf = DecisionTreeClassifier(random_state=42)

    for train_index, test_index in kf.split(X_resampled, y_resampled):
        # 使用 train_index 来获取训练集的索引
        x_train, x_val = features.iloc[train_index], features.iloc[test_index]
        y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_val)

        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        report = classification_report(y_val, y_pred)
        print("Classification Report:")
        print(report)
