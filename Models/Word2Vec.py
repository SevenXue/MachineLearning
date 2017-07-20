# -*- coding:utf-8 -*-


from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
x, y = news.data, news.target

from bs4 import BeautifulSoup
import nltk, re


