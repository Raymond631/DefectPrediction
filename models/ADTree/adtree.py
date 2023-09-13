import math

import joblib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, \
    auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from models.mlp.mlp import plot, data_process


def ad_tree():
    # 指定目标目录的路径
    directory_path = '../../data/csv/MDP/D1/PC5.csv'  # 替换成你的目录路径
    features, labels = data_process(directory_path)

    # 使用随机欠采样
    rus = RandomUnderSampler(sampling_strategy=1, random_state=0, replacement=True)
    # X_resampled, y_resampled = rus.fit_resample(features, labels)
    X_resampled, y_resampled = features, labels
    # 定义一个决策树分类器
    clf = DecisionTreeClassifier(
        criterion='gini',
        splitter='random',
        max_depth=100,
        min_samples_split=2,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.03,
        max_features=40,
        max_leaf_nodes=50,
        class_weight='balanced',
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "../../files/adtree.pkl")
    y_pred = clf.predict(X_val)
    y_score = clf.predict_proba(X_val)
    y_score = y_score[:, 1]
    report = classification_report(y_val, y_pred)
    print("分类报告：")
    print(report)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1score = f1_score(y_val, y_pred, average='weighted')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    false_positive_rate, true_positive_rate = false_positive_rate[1], true_positive_rate[1]
    g_mean = math.sqrt(true_positive_rate * (1 - false_positive_rate))
    balance = 1 - math.sqrt(math.pow((1 - true_positive_rate), 2) + math.pow((0 - false_positive_rate), 2)) / math.sqrt(2)
    print('准确率:', accuracy)  # 准确率
    print('精确率:', precision)  # 精确率
    print('召回率:', recall)  # 召回率
    print('f1值:', f1score)
    # 收敛标准，一般大于0.7时采纳模型
    print('auc_ave:', roc_auc)
    print('g_mean_ave:', g_mean)
    print('balance_ave:', balance)
    plot(y_val, y_score)


if __name__ == '__main__':
    ad_tree()
