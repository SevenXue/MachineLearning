# ~特征筛选
# ~使用Titanic数据，提升决策树性能

import pandas as pd

# 获取数据并预处理
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.info()  #!查看数据的统计信息

#分离数据特征与预测目标
y = titanic['survived']
x = titanic.drop(['row.name', 'name', 'survived'], axis=1)

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
x_test = vec.transform(x_test.to_dict(orinet='record'))

print(len(vec.feature_names_))