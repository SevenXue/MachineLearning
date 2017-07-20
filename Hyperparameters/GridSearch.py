# 使用单线程对文本分类的支持向量机模型的超参数租户执行网格搜索

from sklearn.datasets import fetch_20newsgroups
import numpy as np

news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split

#对前3000条新闻进行数据分割，25%文本用于测试
x_train, x_test, y_train, y_test = train_test_split(news.data[:3000],\
news.target[:3000], test_size=0.25, random_state=33)

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

#导入pipeline
from sklearn.pipeline import Pipeline

#使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])

#2个超参数的个数分别是4，3，svc_gamma的参数共有12
parameters = {'svc__gamma':np.logspace(-2, 1, 4), 'svc__C':np.logspace(-1, 1, 3)}

from sklearn.grid_search import GridSearchCV

#将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知
# 初始化配置并行网格搜索，n_jobs=-1代表使用该计算机全部的CPU
if __name__ == "__main__":
    gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
    gs.fit(x_train, y_train)
    print(gs.best_params_)
    print(gs.best_score_)

    print(gs.score(x_test, y_test))