import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def svm(folder_path):
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
    combined_data.iloc[:, -1] = combined_data.iloc[:, -1].apply(lambda x: 0 if x == b'buggy' else 1)

    # 分割数据为特征 (X) 和目标变量 (y)
    X = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1].astype(int)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建SVM分类器
    svm_model = SVC(kernel='rbf', probability=True)  # 可以用不同的核函数,如'linear'、'rbf'等
    # 训练模型
    svm_model.fit(X_train, y_train)

    # 使用模型进行预测
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

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


if __name__ == '__main__':
    svm('../../data/arff/AEEEM')
