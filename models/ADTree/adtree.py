import joblib
from sklearn.tree import DecisionTreeClassifier

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name


def train_adt(X_train, y_train):
    # 创建adt分类器
    adt_model = DecisionTreeClassifier(
        criterion='gini',
        splitter='random',
        max_depth=100,
        min_samples_split=2,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.03,
        max_features=40,
        max_leaf_nodes=50,
        class_weight='balanced',
        random_state=42,
    )
    # 训练模型
    adt_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(adt_model, '../../out/adtree.pkl')


def test_adt(X_test):
    # 加载模型
    adt_model = joblib.load('../../out/adtree.pkl')
    # 使用模型进行预测
    y_pred = adt_model.predict(X_test)
    y_prob = adt_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def ad_tree(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_adt(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_adt(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"ADTree : {path_dataset_name(folder_path)}")


if __name__ == '__main__':
    ad_tree('../../data/arff/DPDATA', b'buggy')
