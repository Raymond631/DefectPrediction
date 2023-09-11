
import math
import joblib
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, roc_curve, accuracy_score, classification_report, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from models.mlp.mlp import dataset_process, plot, data_process


def multilayer_perceptron():
    directory_path = '../../data/csv/MORPH'
    features,labels = dataset_process(directory_path)
    print(type(features))
    print(type(labels))
    # 使用随机欠采样
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(features, labels)
    #X_resampled, y_resampled =features,labels

    # 定义mlp的分类器
    clf = MLPClassifier(hidden_layer_sizes=(40,80,60,40,20,10,5,2,1), activation='identity', solver='lbfgs', alpha=0.001,
                        batch_size=50, learning_rate='adaptive', learning_rate_init=0.3, power_t=1, max_iter=200,
                        shuffle=True, random_state=42, tol=0.0001, verbose=True, warm_start=True, momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

    x_train, x_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    clf.fit(x_train, y_train)
    joblib.dump(clf, "../../files/mlp.pkl")
    # 使用验证集预测结果
    pre = clf.predict(x_val)
    y_score=clf.predict_proba(x_val)
    y_score = y_score[:, 1]
    # 计算评估指标
    accuracy = accuracy_score(y_val, pre)
    precision = precision_score(y_val, pre, average='weighted')
    recall = recall_score(y_val, pre, average='weighted')
    f1score = f1_score(y_val, pre, average='weighted')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    false_positive_rate, true_positive_rate = false_positive_rate[1], true_positive_rate[1]
    # 几何平均
    g_mean = math.sqrt(true_positive_rate * (1 - false_positive_rate))
    # 平衡度（Balance）是一种用于评估分类模型的平衡性的指标。它的计算方式是根据真阳性率和伪阳性率的欧几里得距离来衡量，除以根号 2。
    balance = 1 - math.sqrt(math.pow((1 - true_positive_rate), 2) + math.pow((0 - false_positive_rate), 2)) / math.sqrt(2)
    print('准确率：', accuracy_score(y_val, pre))
    print('分类报告：', classification_report(y_val, pre))
    plot(y_val,y_score)

if __name__ == '__main__':
    multilayer_perceptron()