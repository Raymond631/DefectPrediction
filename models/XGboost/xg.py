import joblib
from xgboost.sklearn import XGBClassifier

from utils.common import model_evaluation, data_standard_scaler, data_split, read_arff


def train_xgboost(X_train, y_train):
    # 创建并训练XGBoost分类器
    xgboost_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42)
    # 训练模型
    xgboost_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(xgboost_model, '../../files/xgboost.pkl')


def test_xgboost(X_test):
    # 加载模型
    xgboost_model = joblib.load('../../files/xgboost.pkl')
    # 使用模型进行预测
    xgboost_pred = xgboost_model.predict(X_test)
    xgboost_prob = xgboost_model.predict_proba(X_test)[:, 1]
    return xgboost_pred, xgboost_prob


def xgboost(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_xgboost(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_xgboost(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    xgboost('../../data/arff/AEEEM', b'buggy')
