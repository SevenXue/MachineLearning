# ~数据预处理
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.info()  #!查看数据的统计信息

x = titanic[['pclass', 'age', 'sex']]

y = titanic['survived']

# 对缺失数据进行补齐
x['age'].fillna(x['age'].mean(), inplace=True)


# ~建立模型
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# ！特征转换器，将类别型转换为数值型
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)
x_test = vec.fit_transform(x_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

y_predict = dtc.predict(x_test)

# ~性能测试
from sklearn.metrics import classification_report

print(dtc.score(x_test, y_test))

print(classification_report(y_predict, y_test, target_names=['died', 'survived']))
