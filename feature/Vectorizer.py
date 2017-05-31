'''
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'Lobdon', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())
'''

# ~ CountVectorizer对文本数据进行处理
# ~ Naive Bayes + 新闻文本数据

# ~ 未剔除Stop Words 对比CountVectorizer + TfidfVectorizer
# 数据预处理
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# CountVectorizer文本处理
from sklearn.feature_extraction.text import CountVectorizer

# 默认配置，不去除STOP WORDS,
count_ver = CountVectorizer()

x_count_train = count_ver.fit_transform(x_train)
x_count_test = count_ver.transform(x_test)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB

mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)

y_count_predict = mnb_count.predict(x_count_test)

# 模型性能分析
print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer without filtering stopwords):',
      mnb_count.score(x_count_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_count_predict, target_names=news.target_names))

#TfidfVectorizer 文本处理
from sklearn.feature_extraction.text import TfidfVectorizer

#默认配置，未去除StopWords
tfidf_vec = TfidfVectorizer()

x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

# Naive Bayes
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train, y_train)

y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)

# 性能分析
print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(x_tfidf_test, y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))


# ~去掉StopWords Count VS Tfidf

#过滤词为english
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

#对测试样本进行量化处理
x_count_filter_train = count_filter_vec.fit_transform(x_train)
x_count_filter_test = count_filter_vec.transform(x_test)

x_tfidf_filter_train = tfidf_filter_vec.fit_transform(x_train)
x_tfidf_filter_test = tfidf_filter_vec.transform(x_test)

#Naive Bayes
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(x_count_filter_train, y_train)
y_count_filter_predict = mnb_count_filter.predict(x_count_filter_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(x_tfidf_filter_train, y_train)
y_tfidf_filter_predict = mnb_tfidf_filter.predict(x_count_filter_test)

#性能分析
print('The accuracy of classifying 20newsgroups using Naive Bayes(CountVectorizer by filtering stopwords):',
      mnb_count_filter.score(x_count_filter_test, y_test))
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))

print('The accuracy of classifying 20newsgroups using Naive Bayes(TfidfVectorizer by filtering stopwords):',
      mnb_tfidf_filter.score(x_tfidf_filter_test, y_test))
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))
