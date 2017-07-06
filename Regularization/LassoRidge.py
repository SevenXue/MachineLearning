#~ L1,L2范数正则化,
#~ 使用的数据为pizza价格

x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

#~ 通过图形化，比较2次和4次多项式的差异
from sklearn.linear_model import LinearRegression
#导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

#degree=4 表示初始化2,4次多项式特征生成器
poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train)

poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)

#线性回归
regress_poly2 = LinearRegression()
regress_poly2.fit(x_train_poly2, y_train)

regress_poly4 = LinearRegression()
regress_poly4.fit(x_train_poly4, y_train)

x_test_poly2 = poly2.transform(x_test)
x_test_poly4 = poly4.transform(x_test)
print(regress_poly2.score(x_test_poly2, y_test))
print(regress_poly4.score(x_test_poly4, y_test))


#~ 使用不同的正则化方式
#Lasso
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
print(lasso_poly4.score(x_test_poly4, y_test))
#Ridge
from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print(ridge_poly4.score(x_test_poly4, y_test))

#绘图
import numpy as np
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
# print(xx.shape)
xx_poly2 = poly2.transform(xx)
xx_poly4 = poly4.transform(xx)
yy_poly2 = regress_poly2.predict(xx_poly2)
yy_poly4 = regress_poly4.predict(xx_poly4)

import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)
plt2, = plt.plot(xx, yy_poly2, label='Degree2')
plt4, = plt.plot(xx, yy_poly4, label='Degree4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt2, plt4])
plt.show()


