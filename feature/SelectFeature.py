# ~特征筛选
# ~使用Titanic数据，提升决策树性能

import pandas as pd

# 获取数据并预处理
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.info()  #!查看数据的统计信息

#分离数据特征与预测目标
y = titanic['survived']
x = titanic.drop(['row.names', 'name', 'survived'], axis=1)

#对缺失数据进行填充
x['age'].fillna(x['age'].mean(), inplace=True)
x.fillna('UNKNOW', inplace=True)

# 分割数据
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x_train =  vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

#print(len(vec.feature_names_))

#使用决策树模型进行预测
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test)) #0.8115

#从sklearn导入特征筛选器
from sklearn import feature_selection

'''
#筛选前20%的特征
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
x_train_fs = fs.fit_transform(x_train, y_train)
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dt.score(x_test_fs, y_test))
'''

#通过交叉验证，按照固定间隔的百分比筛选特征，并作图展示
from sklearn.cross_validation import cross_val_score
import numpy as np

percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    results.append(scores.mean())
print(results)

import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

#使用最佳筛选后的特征，进行测试
fsbest = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
x_train_fsbest = fsbest.fit_transform(x_train, y_train)
dt.fit(x_train_fsbest, y_train)
x_test_fsbest = fsbest.transform(x_test)
dt.score(x_test_fsbest, y_test)
