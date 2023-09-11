from sklearn.svm import SVC

from utils.common import model_evaluation, read_arff, data_split, data_standard_scaler


def svm(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 创建SVM分类器
    svm_model = SVC(kernel='rbf', probability=True)  # 可以用不同的核函数,如'linear'、'rbf'等
    # 训练模型
    svm_model.fit(X_train, y_train)
    # 使用模型进行预测
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    svm('../../data/arff/AEEEM', b'buggy')
