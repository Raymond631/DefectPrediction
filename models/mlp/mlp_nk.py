import joblib
from sklearn.neural_network import MLPClassifier

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name


def train_mlp(X_train, y_train):
    # 创建多层感知器
    clf = MLPClassifier(hidden_layer_sizes=(40, 80, 60, 40, 20, 10, 5, 2, 1), activation='tanh', solver='lbfgs',
                        alpha=0.001, batch_size=50, learning_rate='adaptive', learning_rate_init=0.03, power_t=0.5,
                        max_iter=200,
                        shuffle=True, random_state=42, tol=0.0001, verbose=True, warm_start=True, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
    # 训练模型
    for i in range(2):
        clf.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(clf, "../../out/mlp.pkl")


def test_mlp(X_test):
    # 加载模型
    clf = joblib.load('../../out/mlp.pkl')
    # 使用模型进行预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def multilayer_perceptron(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_mlp(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_mlp(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"MLP : {path_dataset_name(folder_path)}")


if __name__ == '__main__':
    multilayer_perceptron('../../data/arff/AEEEM', b'buggy')
