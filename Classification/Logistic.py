# 逻辑回归
# 癌症数据集

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names = column_names)

## 数据预处理
data = data.replace(to_replace='?', value=np.nan)   #将？替换为np.nan
data = data.dropna(how='any')   #丢弃带有缺失值的数据！

x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                     test_size=0.25, random_state=33)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

## 通过两种模型进行预测
# 标准化数据，保证每个维度的特征数据的方差为1，均值为0
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 初始化LogisticRegression和SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

#调用fit函数训练模型参数
lr.fit(x_train, y_train)
#调用模型lr对x_test进行预测
lr_y_predict = lr.predict(x_test)

sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)


## 性能分析
from sklearn.metrics import classification_report

print("Accuracy of LR Classifier:", lr.score(x_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Begin', 'Malignant']))


'''
print(y_train.value_counts())
print(y_test.value_counts())
'''