import joblib
from sklearn.naive_bayes import GaussianNB

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation


def train_nb(X_train, y_train):
    # 创建NB分类器
    nb_model = GaussianNB(var_smoothing=1e-9)
    # 训练模型
    nb_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(nb_model, '../../files/nb.pkl')


def test_nb(X_test):
    # 加载模型
    nb_model = joblib.load('../../files/nb.pkl')
    # 使用模型进行预测
    y_pred = nb_model.predict(X_test)
    y_prob = nb_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def naive_bayes(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_nb(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_nb(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    print('nb')
    naive_bayes('../../data/arff/AEEEM', b'buggy')
