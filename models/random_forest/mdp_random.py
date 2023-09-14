import joblib
from sklearn.ensemble import RandomForestClassifier

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name


def train_rf(X_train, y_train):
    # 创建随机森林分类器
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    # 训练模型
    clf.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(clf, "../../out/rf.pkl")


def test_rf(X_test):
    # 加载模型
    svm_model = joblib.load('../../out/rf.pkl')
    # 使用模型进行预测
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def rf(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_rf(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_rf(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"Random forest : {path_dataset_name(folder_path)}")


if __name__ == '__main__':
    rf('../../data/arff/AEEEM', b'buggy')
