#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:08:04 2018

@author: carlocarandang
"""

import nltk
import random
from re import split
from nltk.corpus import stopwords
documents = []

# Dataset contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants
read_file = "/Users/carlocarandang/Desktop/Data Mining/NLP-Sentiment Analysis/sentiment labelled sentences/amazon_cells_labelled.txt"

# Format of dataset: sentence \t score \n

with open(read_file,"r") as r:
    c=0
    for line in r:
        #create the training document of labeled tuples
        splitted = line.strip().split('\t')
        msg = (' ').join(splitted[:-1])
        is_class = splitted[-1]
        documents.extend([ dict(doc=msg.lower(), category=is_class)]) #add a dict value which is all the words/tokens
        
for n in range(len(documents)):
    documents[n]['words'] = split('\W+', documents[n]['doc'])
#take out stopwords and any other garbage like numbers and all
all_words = nltk.FreqDist(w.lower() for d in documents for w in d['words'] if w not in stopwords.words() and not w.isdigit())
#experiment with different amounts of features
word_features = all_words.keys()[:2500]

#get features labeled with true or false (words in this case)
def document_features(document):
    #this if statement is only needed if you’re using a saved classifier
    if not document.get('words'):
        document['words'] = split('\W+', document['doc']) #split into words
    document_words = set(document['words']) #unique them
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

random.shuffle(documents)
#get featuresets for each document
featuresets = [(document_features(d), d['category']) for d in documents]
#split training and testing sets (try smaller and bigger or more balances training sets)
train_set, test_set = featuresets[:800], featuresets[800:]

#classify using in-built Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
print "-------------"
print "The accuracy of the Naive Bayes Classifier is: ", nltk.classify.accuracy(classifier, test_set) 
print "-------------"
classifier.show_most_informative_features(25)
print "-------------------"
print "Testing the review, 'does not work': ", classifier.classify(document_features({'doc':"does not work"}))
print "-------------------"
print "Testing the review, 'works well': ", classifier.classify(document_features({'doc':"works well"}))
