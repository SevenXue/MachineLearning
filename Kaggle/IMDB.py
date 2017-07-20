# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

train = pd.read_csv('../Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../Datasets/IMDB/labeledTrainData.tsv', delimiter='\t')

#查看数据的信息
#print(train.info())
#查看数据的前几条信息
#print(train.head())


from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk


def review_to_text(review, remove_stopwords):
#任务1：去掉html标记
    raw_text = BeautifulSoup(review,'html').get_text()
#任务2：去掉非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
#任务：如果remove_stopwords被激活，则进一步去掉评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]

    return words


#分别对原始训练和测试数据集进行上述三项预处理
x_train = []
for review in train['review']:
    x_train.append(' '.join(review_to_text(review, True)))
x_test = []
for review in test['review']:
    x_test.append(' '.join(review_to_text(review, True)))

y_train = train['sentiment']

#导入文本特征提取器CountVectorizer与TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


#使用pipeline搭建两组使用朴素贝叶斯模型的分类器， 区别在于使用CountVectorizer和TfidfVectorizer对文本进行提取
pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

pip_tfidf = Pipeline([('tfidf', TfidfVectorizer(analyzer='word')), ('mnb', MultinomialNB())])

#分别配置用于模型超参数搜索组合
params_count = {'count_vec__binary':[True, False], 'count_vec__ngram_range':[(1, 1), (1, 2)],
                'mnb__alpha':[0.1, 1.0, 10.0]}
params_tfidf = {'tfidf__binary':[True, False], 'tfidf__ngram_range':[(1, 1), (1, 2)],
                'mnb__alpha':[0.1, 1.0, 10.0]}

#采用4折交叉验证的方法对使用CountVectorizer的朴素贝叶斯模型进行并行超参数搜索

if __name__ == '__main__':
    gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
    gs_count.fit(x_train, y_train)
    #输出交叉验证中最佳的准确性得分和超参数组合
    print(gs_count.best_score_)
    print(gs_count.best_params_)

    count_y_predict = gs_count.predict(x_test)

    gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
    gs_tfidf.fit(x_train, y_train)
    #输出最佳结果
    print(gs_tfidf.best_score_)
    print(gs_tfidf.best_params_)

    tfidf_y_predict = gs_tfidf.predict(x_test)

    #使用pandas对数据进行格式化
    submission_count = pd.DataFrame({'id':test['id'], 'sentimet': count_y_predict})
    submission_tfidf = pd.DataFrame({'id':test['id'], 'sentimet': tfidf_y_predict})

    #结果输出到本地
    submission_count.to_csv('../Datasets/IMDB/submission_count.csv', index=False)
    submission_tfidf.to_csv('../Datasets/IMDB/submission_tfidf.csv', index=False)


