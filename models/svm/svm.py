import os

import pandas as pd
from scipy.io import arff
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def svm(folder_path):
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

    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建SVM分类器
    svm_classifier = SVC(kernel='rbf')  # 可以用不同的核函数,如'linear'、'rbf'等

    # 训练模型
    svm_classifier.fit(X_train, y_train)

    # 使用模型进行预测
    y_pred = svm_classifier.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # 打印分类报告
    print(classification_report(y_test, y_pred, target_names=class_labels))


if __name__ == '__main__':
    svm('../../data/arff/AEEEM')
    # data_path = "../../data"
    # # 所有数据集
    # data_sets = [f for f in os.listdir(data_path)]
    # # 分别使用各个数据集
    # for data_set in data_sets:
    #     print(f'数据集：{data_set}')
    #     svm(os.path.join(data_path, data_set))
    #     print('-----------------------------')
