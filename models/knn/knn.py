import joblib
from sklearn.neighbors import KNeighborsClassifier

from utils.common import read_arff, data_standard_scaler, data_split, model_evaluation


def train_knn(X_train, y_train):
    # 创建knn分类器
    knn_model = KNeighborsClassifier(n_neighbors=5)
    # 训练模型
    knn_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(knn_model, '../../out/knn.pkl')


def test_knn(X_test):
    # 加载模型
    knn_model = joblib.load('../../out/svm.pkl')
    # 使用模型进行预测
    y_pred = knn_model.predict(X_test)
    y_prob = knn_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def knn(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_knn(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_knn(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    knn('../../data/arff/AEEEM', b'buggy')
