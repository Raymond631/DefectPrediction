from models.svm.svm import train_svm
from models.svm.test_svm import test_svm
from utils.common import read_arff, data_split, data_standard_scaler


def base_classifier():
    pass


def meta_classifier():
    pass


if __name__ == '__main__':
    # 读取arff数据集
    df = read_arff('../../data/arff/AEEEM', b'buggy')
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_svm(X_train, y_train)
    # 预测
    X = df.iloc[:, :-1].values
    y_pred, y_prob = test_svm(X)

    # 将预测结果插入数据集
    df.insert(loc=0, column='pred', value=y_pred)
    # 将数据分割为训练集和测试集
    P_train, P_test, q_train, q_test = data_split(df)
