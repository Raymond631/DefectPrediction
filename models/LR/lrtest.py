# # 任务：是否患有糖尿病（二分类）
# 模型：LR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 切分数据集
df = pd.read_csv("../../data/csv/AEEEM/EQ.csv")
target = df.pop("class")
data = df.values
X = data
Y = target
Y = Y.replace({'buggy': 0, 'clean': 1})
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)

# LR模型预测
lr = LogisticRegression()  #初始化LogisticRegression
lr.fit(X_train, Y_train)  # 调用LogisticRegression中的fit函数训练模型参数
lr_pres = lr.predict(X_test) # 使用训练好的模型lr对X_test进行预测
print('准确率：',accuracy_score(Y_test, lr_pres))
print('精确率：',precision_score(Y_test, lr_pres))
print('召回率：',recall_score(Y_test, lr_pres))
print("Classification Report:")
print(classification_report(Y_test, lr_pres, target_names=['buggy', 'clean']))


# ROC曲线和AUC
lr_pres_proba = lr.predict_proba(X_test) [::, 1]
fpr, tpr, thresholds = roc_curve(Y_test, lr_pres_proba)
auc = roc_auc_score(Y_test, lr_pres_proba)
plt.figure(figsize=(5, 3), dpi=100)
plt.plot(fpr, tpr, label="AUC={:.2f}" .format(auc))
plt.legend(loc=4, fontsize=10)
plt.title('ROC',fontsize=20)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.show()
# 计算测试集中正类别的占比，即"clean"的占比
positive_class_ratio = np.sum(Y_test == 1) / len(Y_test)



# 绘制ROC曲线和AUC
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(fpr, tpr, label="AUC={:.2f}".format(auc))

# 绘制从(0,0)到(1,1)的直线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label="Random Guess")

plt.legend(loc=4, fontsize=10)
plt.title('ROC', fontsize=20)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()






