import joblib
from sklearn.svm import SVC

from models.svm.test_svm import test_svm
from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation


def train_svm(X_train, y_train):
    # 创建SVM分类器
    svm_model = SVC(kernel='rbf', probability=True)  # 可以用不同的核函数,如'linear'、'rbf'等
    # 训练模型
    svm_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(svm_model, '../../files/svm.pkl')


if __name__ == '__main__':
    # 读取arff数据集
    df = read_arff('../../data/arff/AEEEM', b'buggy')
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_svm(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_svm(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)
