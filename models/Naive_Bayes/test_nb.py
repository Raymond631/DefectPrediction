from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from models.Naive_Bayes.model_nb import data_pre_processing, naive_Bayes
from models.Naive_Bayes.model_nb import plot_roc
from models.svm.svm import svm

if __name__ == '__main__':
    print('组合')
    X, y, class_labels = data_pre_processing('../../data/arff/AEEEM')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 定义基分类器
    classifier1 = GaussianNB(var_smoothing=1e-9)
    classifier2 = SVC(kernel='rbf', probability=True)
    classifier1.fit(X_train, y_train)
    y_proba_model1 = classifier1.predict_proba(X_test)[:, 1]
    classifier2.fit(X_train, y_train)
    y_proba_model2 = classifier2.predict_proba(X_test)[:, 1]

    # 定义组合训练器
    ensemble_classifier = VotingClassifier(estimators=[('svm', classifier2), ('nb', classifier1)])
    ensemble_classifier.fit(X_train, y_train)
    y_pred = ensemble_classifier.predict(X_test)
    auc = metrics.accuracy_score(y_test, y_pred)
    macro = metrics.precision_score(y_test, y_pred, average='macro')
    micro = metrics.precision_score(y_test, y_pred, average='micro')
    macro_recall = metrics.recall_score(y_test, y_pred, average='macro')
    weighted = metrics.f1_score(y_test, y_pred, average='weighted')

    # 对每个样本，计算所有模型的预测概率的平均值作为最终预测结果
    y_proba_combined = (y_proba_model1 + y_proba_model2) / 2
    print('分类报告：', classification_report(y_test, y_pred, target_names=class_labels))
    plot_roc(y_test, y_proba_combined, auc, macro, macro_recall, weighted)  # 绘制ROC曲线并求出AUC值
    print('nb')
    naive_Bayes('../../data/arff/AEEEM')
    print('svm')
    svm('../../data/arff/AEEEM')
