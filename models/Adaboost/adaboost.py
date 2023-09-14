import joblib
# 模型：LR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name


def train_adaboost(X_train, y_train):
    base_classifier = LogisticRegression()  # 选择LR的弱分类器
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=42)
    # 训练AdaBoost分类器
    adaboost_classifier.fit(X_train, y_train)
    joblib.dump(adaboost_classifier, '../../out/adaboost.pkl')


def test_adaboost(X_test):
    # 加载模型
    adaboost_model = joblib.load('../../out/adaboost.pkl')
    # 使用模型进行预测
    adaboost_pred = adaboost_model.predict(X_test)
    adaboost_prob = adaboost_model.predict_proba(X_test)[:, 1]
    return adaboost_pred, adaboost_prob


def adaboost(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_adaboost(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_adaboost(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"Adaboost : {path_dataset_name(folder_path)}")


if __name__ == '__main__':
    adaboost('../../data/arff/AEEEM', b'buggy')
