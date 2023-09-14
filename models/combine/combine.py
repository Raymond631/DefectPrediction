# 引入模型
from models.ADTree.adtree import train_adt, test_adt
from models.Adaboost.adaboost import train_adaboost, test_adaboost
from models.DT.dt import train_dt, test_dt
from models.LR.lrtest import train_lr, test_lr
from models.Naive_Bayes.model_nb import train_nb, test_nb
from models.XGboost.xg import train_xgboost, test_xgboost
from models.knn.knn import train_knn, test_knn
from models.mlp.mlp_nk import train_mlp, test_mlp
from models.random_forest.mdp_random import train_rf, test_rf
from models.svm.svm import train_svm, test_svm
from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name

df = ''


def base_classifier(X_train, y_train, X, model):
    global df
    if model == 'naive_bayes':
        # 训练模型
        train_nb(X_train, y_train)
        # 预测
        y_pred, y_prob = test_nb(X)
    elif model == 'svm':
        train_svm(X_train, y_train)
        y_pred, y_prob = test_svm(X)
    elif model == 'ADTree':
        train_adt(X_train, y_train)
        y_pred, y_prob = test_adt(X)
    elif model == 'dt':
        train_dt(X_train, y_train)
        y_pred, y_prob = test_dt(X)
    elif model == 'lr':
        train_lr(X_train, y_train)
        y_pred, y_prob = test_lr(X)
    elif model == 'mlp':
        train_mlp(X_train, y_train)
        y_pred, y_prob = test_mlp(X)
    elif model == 'adaboost':
        train_adaboost(X_train, y_train)
        y_pred, y_prob = test_adaboost(X)
    elif model == 'xgboost':
        train_xgboost(X_train, y_train)
        y_pred, y_prob = test_xgboost(X)
    elif model == 'random_forest':
        train_rf(X_train, y_train)
        y_pred, y_prob = test_rf(X)
    elif model == 'knn':
        train_knn(X_train, y_train)
        y_pred, y_prob = test_knn(X)
    else:
        y_pred, y_prob = test_knn(X)

    # 将预测结果插入数据集
    df.insert(loc=0, column='pred', value=y_pred)
    # 将数据分割为训练集和测试集
    P_train, P_test, q_train, q_test = data_split(df)

    return P_train, P_test, q_train, q_test


def meta_classifier(X_train, y_train, X_test, model):
    if model == 'naive_bayes':
        train_nb(X_train, y_train)
        y_pred, y_prob = test_nb(X_test)
    elif model == 'svm':
        train_svm(X_train, y_train)
        y_pred, y_prob = test_svm(X_test)
    elif model == 'ADTree':
        train_adt(X_train, y_train)
        y_pred, y_prob = test_adt(X_test)
    elif model == 'dt':
        train_dt(X_train, y_train)
        y_pred, y_prob = test_dt(X_test)
    elif model == 'lr':
        train_lr(X_train, y_train)
        y_pred, y_prob = test_lr(X_test)
    elif model == 'mlp':
        train_mlp(X_train, y_train)
        y_pred, y_prob = test_mlp(X_test)
    elif model == 'adaboost':
        train_adaboost(X_train, y_train)
        y_pred, y_prob = test_adaboost(X_test)
    elif model == 'xgboost':
        train_xgboost(X_train, y_train)
        y_pred, y_prob = test_xgboost(X_test)
    elif model == 'random_forest':
        train_rf(X_train, y_train)
        y_pred, y_prob = test_rf(X_test)
    elif model == 'knn':
        train_knn(X_train, y_train)
        y_pred, y_prob = test_knn(X_test)
    else:
        y_pred, y_prob = test_knn(X)
    return y_pred, y_prob


def combine_models(folder_path, bug_label, models):
    global df
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 特征变量
    X = df.iloc[:, :-1].values
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 基分类器
    for i in range(len(models) - 1):
        X_train, X_test, y_train, y_test = base_classifier(X_train, y_train, X, models[i])

    # 元分类器
    y_pred, y_prob = meta_classifier(X_train, y_train, X_test, models[-1])
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"Combine : {path_dataset_name(folder_path)}")
