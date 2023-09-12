from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils.common import read_arff, model_evaluation


def naive_bayes(folder_path, bug_label):
    combined_data = read_arff(folder_path, bug_label)
    # 分割数据为特征 (X) 和目标变量 (y)
    X = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练分类器
    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # 模型评估
    model_evaluation(y_test, y_pred, y_prob)


if __name__ == '__main__':
    print('nb')
    naive_bayes('../../data/arff/AEEEM', b'buggy')
