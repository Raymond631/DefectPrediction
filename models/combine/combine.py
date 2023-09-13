from sklearn.neural_network import MLPClassifier

from models.svm.svm import train_svm
from models.svm.test_svm import test_svm
from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation


def base_classifier(X_train, y_train, X):
    # 训练模型
    train_svm(X_train, y_train)
    # 预测
    y_pred, y_prob = test_svm(X)

    # 将预测结果插入数据集
    df.insert(loc=0, column='pred', value=y_pred)
    # 将数据分割为训练集和测试集
    P_train, P_test, q_train, q_test = data_split(df)

    return P_train, P_test, q_train, q_test


def meta_classifier(X_train, y_train, X_test):
    clf = MLPClassifier(hidden_layer_sizes=(40, 80, 60, 40, 20, 10, 5, 2, 1), activation='tanh', solver='sgd',
                        alpha=0.001, batch_size=50, learning_rate='adaptive', learning_rate_init=0.03, power_t=0.5, max_iter=200,
                        shuffle=True, random_state=42, tol=0.0001, verbose=True, warm_start=True, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

    for i in range(1):
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


if __name__ == '__main__':
    # 读取arff数据集
    df = read_arff('../../data/arff/AEEEM', b'buggy')
    # 特征变量
    X = df.iloc[:, :-1].values
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 基分类器：SVM
    P_train, P_test, q_train, q_test = base_classifier(X_train, y_train, X)
    # 元分类器：MLP
    q_pred, q_prob = meta_classifier(P_train, q_train, P_test)
    # 模型评估
    model_evaluation(q_test, q_pred, q_prob)
