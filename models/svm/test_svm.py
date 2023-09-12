import joblib


def test_svm(X_test):
    # 加载模型
    svm_model = joblib.load('../../files/svm.pkl')
    # 使用模型进行预测
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob
