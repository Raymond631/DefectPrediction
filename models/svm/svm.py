from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.common import model_evaluation, read_arff


def svm(folder_path, bug_label):
    # 读取arff数据集
    combined_data = read_arff(folder_path, bug_label)

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

    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    svm('../../data/arff/AEEEM', b'buggy')
