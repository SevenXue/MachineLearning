# 线性回归，SVM核函数回归
# 波士顿地区房价数据


from sklearn.datasets import load_boston
boston = load_boston()

from sklearn.cross_validation import train_test_split
import numpy as np

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.25)
print("The max target value is", np.max(boston.target))
print("The min target value is", np.min(boston.target))
print("The average target value is", np.mean(boston.target))

from sklearn.preprocessing import StandardScaler

ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor()

sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)

# ~性能分析
print('The value of defalut measurement of LinearRegression is', lr.score(x_test, y_test))

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
print('The value of R-squred of LinearRegression is', r2_score(y_test, lr_y_predict))
print('The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(lr_y_predict)))
print('The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(lr_y_predict)))


print('The value of defalut measurement of SGDRegression is', sgdr.score(x_test, y_test))

print('The value of R-squred of SGDRegression is', r2_score(y_test, sgdr_y_predict))
print('The mean squared error of SGDRegression is', mean_squared_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(sgdr_y_predict)))
print('The mean absolute error of SGDRegression is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                          ss_y.inverse_transform(sgdr_y_predict)))


from sklearn.svm import SVR

linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('R-squared value of linear SVR is', r2_score(y_test, linear_svr_y_predict))
print("The mean squared error of linear VSR is", mean_squared_error(ss_y.inverse_transform(y_test),
                                                                    ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear VSR is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                      ss_y.inverse_transform(linear_svr_y_predict)))

print('R-squared value of poly SVR is', r2_score(y_test, poly_svr_y_predict))
print("The mean squared error of poly VSR is", mean_squared_error(ss_y.inverse_transform(y_test),
                                                                    ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of poly VSR is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                      ss_y.inverse_transform(poly_svr_y_predict)))

print('R-squared value of rbf SVR is', r2_score(y_test, rbf_svr_y_predict))
print("The mean squared error of rbf VSR is", mean_squared_error(ss_y.inverse_transform(y_test),
                                                                    ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absolute error of rbf VSR is', mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                      ss_y.inverse_transform(rbf_svr_y_predict)))