import joblib
from sklearn.tree import DecisionTreeClassifier

from utils.common import read_arff, data_split, data_standard_scaler, model_evaluation, path_dataset_name


def train_dt(X_train, y_train):
    # 创建dt分类器
    dt_model = DecisionTreeClassifier(
        criterion='gini',  # 不纯度度量，可选 'gini' 或 'entropy'
        splitter='random',  # 分割策略，可选 'best' 或 'random'
        max_depth=100,  # 树的最大深度，None 表示不限制深度
        min_samples_split=2,  # 节点分割的最小样本数
        min_samples_leaf=2,  # 叶节点的最小样本数
        min_weight_fraction_leaf=0.03,  # 叶节点的最小样本权重总和
        max_features=None,  # 每次分割考虑的最大特征数
        max_leaf_nodes=None,  # 叶节点的最大数量，None 表示不限制数量
        class_weight='balanced',  # 类别权重，None 表示不考虑类别权重
        random_state=42,  # 随机种子，用于重复性
    )
    # 训练模型
    dt_model.fit(X_train, y_train)
    # 保存模型到磁盘
    joblib.dump(dt_model, '../../out/dt.pkl')


def test_dt(X_test):
    # 加载模型
    dt_model = joblib.load('../../out/dt.pkl')
    # 使用模型进行预测
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def decision_tree(folder_path, bug_label):
    # 读取arff数据集
    df = read_arff(folder_path, bug_label)
    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = data_split(df)
    # 标准化特征数据
    X_train, X_test = data_standard_scaler(X_train, X_test)

    # 训练模型
    train_dt(X_train, y_train)
    # 测试模型
    y_pred, y_prob = test_dt(X_test)
    # 模型评估
    model_evaluation(y_test, y_pred, y_prob, f"DT : {path_dataset_name(folder_path)}")


if __name__ == '__main__':
    decision_tree('../../data/arff/DPDATA', b'buggy')
