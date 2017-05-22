# 随机森林
# 使用titanic幸存者数据


# ~数据预处理
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# titanic.info()    #!查看数据的统计信息

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
#print(vec.feature_names_)
x_test = vec.fit_transform(x_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)


# 使用随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)


# 使用梯度上升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

# ~性能分析
from sklearn.metrics import classification_report

#单一决策树
print('The Accuracy of decision tree is :', dtc.score(x_test, y_test))
print(classification_report(y_predict, y_test))

#随机森林
print('The Accuracy of Random Forest is :', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

#梯度提升决策树
print('The Accuracy of Gradient tree boosting is :', gbc.score(x_test, y_test))
print(classification_report(gbc_y_pred, y_test))