# -*- coding:utf-8 -*-
import pandas as pd

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
print(x_train['Embarked'].value_counts())
print(x_test['Embarked'].value_counts())

# 使用出现频率最高的特征来填充，相对可以减少误差
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

#对于Age数值型的特征，填充均值来补齐缺失值
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

print(x_train.info())