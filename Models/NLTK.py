# -*- coding:utf-8 -*-
import nltk

sent1 = 'The cat is walking in the bedroom'
sent2 = 'A dog was running across the kitchen'

tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)

tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)

vocab1 = sorted(tokens_1)
print(vocab1)

vocab2 = sorted(tokens_2)
print(vocab2)

stemmer = nltk.stem.PorterStemmer()
