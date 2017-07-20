# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

#读取测试集和训练集
train = pd.read_csv('../Datasets/titanic/train.csv')
test = pd.read_csv('../Datasets/titanic/test.csv')

#print(train.info())
#print(test.info())

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

x_train = train[selected_features]
x_test = test[selected_features]

y_train = train['Survived']

#查看Embarker特征的数据，补完缺失值
# print(x_train['Embarked'].value_counts())
# print(x_test['Embarked'].value_counts())

# 使用出现频率最高的特征来填充，相对可以减少误差
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

#对于Age数值型的特征，填充均值来补齐缺失值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

# print(x_train.info())

#采用dictVectorizer对特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

#导入RandomForestClassifier
from sklearn.ensemble import  RandomForestClassifier
rfc = RandomForestClassifier()

from xgboost import XGBClassifier
xgbc = XGBClassifier()

#使用5折交叉验证
from sklearn.cross_validation import cross_val_score
cross_val_score(rfc, x_train, y_train, cv=5).mean()
cross_val_score(xgbc, x_train, y_train, cv=5).mean()

rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_predict})
#存储文件
rfc_submission.to_csv('../Datasets/titanic/rfc_submission.csv', index=False)

#使用默认配置的XGBClassifier进行预测操作
xgbc.fit(x_train, y_train)

xgbc_y_predict = xgbc.predict(x_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Surviver':xgbc_y_predict})
xgbc_submission.to_csv('../Datasets/titanic/xgbc_submission.csv', index=False)

# 使用并行网格搜索的方式寻找更好的超参数组合
from sklearn.grid_search import GridSearchCV
parameters = {'max_depth': np.arange(2,7), 'n_estimators':np.arange(100, 1100, 200),
          'learning_rate':np.array([0.05, 0.1, 0.25, 0.5, 1.0])}

if __name__ == "__main__":
    xgbc_best = XGBClassifier()
    gs = GridSearchCV(xgbc_best, parameters, n_jobs=-1, cv=5, verbose=1)
    gs.fit(x_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    xgbc_best_y_prdict = gs.predict(x_test)
    xgbc_best_submission = pd.DataFrame({'PassengerId':test['PassengerId'],
                                         'Survived':xgbc_best_y_prdict})
    xgbc_best_submission.to_csv('../Datasets/titanic/xgbc_best_submission.csv')