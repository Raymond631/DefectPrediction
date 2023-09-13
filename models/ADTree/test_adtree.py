from collections import Counter
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def test_adtree(X_test):
    clf = joblib.load("../../files/adtree.pkl")
    # 使用模型进行预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def adtree_result(X_test):
    clf = joblib.load("../../files/adtree.pkl")
    pre = clf.predict(X_test)
    Counter(pre)
    Yes = sum(pre == 1)
    No = sum(pre == 0)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(6, 6))
    label = ['有缺陷数', '无缺陷数']
    explode = [0.01, 0.05]
    values = [Yes, No]
    plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')  # 绘制饼图
    plt.title('缺陷数目')
    plt.show()

