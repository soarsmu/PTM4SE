#Importing required librariesimport pandas as pd
import typing_extensions
from pandas.tseries.offsets import SemiMonthBegin
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd 
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import random 
from datetime import date
import jieba
import numpy as np
import pandas as pd
import re
import time
import random
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def getdata_undersample(data,label,aspect):
    # make the label with target aspect as 1, without as 0
    # randomly select same number data from dataset0 equals to dataset0

    label_all = []
    data_label1 = []
    data_label0 = []

    # divided positive samples and negative samples
    for i in range(len(label)):
        if aspect in label[i]:
            data_label1.append(data[i])
        else:
            data_label0.append(data[i])

    # randomly select data points from negative samples, number equals to length of positive samples
    # if len(data_label1)<=len(data_label0):
    extracted_data0 = random.sample(data_label0,len(data_label1))
    # else:
    #     # not realistic
    #     extracted_label0 = random.sample(data_label0,len(data_label1))

    # make the label list
    label0 = [0]*len(extracted_data0)
    extracted_data0.extend(data_label1)

    label1 = [1]*len(data_label1)
    label0.extend(label1)

    return extracted_data0,label0


def getdata(data,label,aspect):
    # make the label with target aspect as 1, without as 0
    # randomly select same number data from dataset0 equals to dataset0

    label_all = []


    # divided positive samples and negative samples
    for i in range(len(label)):
        if aspect in label[i]:
            label_all.append(1)
        else:
            label_all.append(0)


    return data,label_all

def train_model(classifier, feature_vector_train, label, feature_vector_valid,valid_y):
# fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    # return metrics.accuracy_score(predictions, valid_y)

    return f1_score(valid_y, predictions),classification_report(valid_y, predictions, labels=[0,1], digits=4)


#Loading the dataset
datapath = '/workspace/pipeline/aspect_classifier/data/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv'
data = pd.read_csv(datapath,sep=',')
X = data['sent'] # data
y = data['codes'] # label 

#Implementing cross validation, shuffle optional
k = 10
kf = StratifiedKFold(n_splits=k, random_state=None)
# kf = StratifiedKFold(n_splits=k, random_state=True,shuffle=True)

f1_all = []

# parameters
aspect = 'Legal'

if aspect == 'Other':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2")
    clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
if aspect == 'Usability':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2",ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)
    clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
if aspect == 'Performance':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2",ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)
    clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
if aspect == 'Security':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english')
    clf =SGDClassifier(alpha=.0001, max_iter=2000, 
                                          epsilon=0.5, loss='log',penalty="l2", 
                                          power_t=0.5, warm_start=False, shuffle=True, class_weight='balanced')
if aspect == 'Community':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2")
    clf = LinearSVC(class_weight="balanced", C=10.0, loss = "squared_hinge", max_iter=1000, penalty="l2")
if aspect == 'Bug':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english')
    clf = SGDClassifier(alpha=.001, max_iter=100, 
                                          epsilon=0.5, loss='squared_hinge',penalty="l2", 
                                          power_t=0.1, warm_start=False, shuffle=True, class_weight='balanced')
if aspect == 'Compatibility':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l1",ngram_range = (1,3), token_pattern = r'\b\w+\b', min_df = 1)
    clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
if aspect == 'Documentation':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2",ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)
    clf = LinearSVC(class_weight="balanced", C=20.0, loss = "squared_hinge", max_iter=1000, penalty="l2")
if aspect == 'Legal':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english')
    clf = SGDClassifier(alpha=.0001, epsilon=0.5, loss='hinge', 
                                  max_iter=50,penalty="l2", power_t=0.5, shuffle=True,warm_start=False, class_weight='balanced')
if aspect == 'Portability':
    tfidf_vect_ngram =  TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english', norm="l2")
    clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")


# training process with 10 folder cross validation
for train_index , test_index in kf.split(X,y):
    
    # get the index of data and label based on aspects
    X_train , X_test = X[train_index].tolist(),X[test_index].tolist()
    y_train , y_test = y[train_index].tolist(), y[test_index].tolist()
    train_data,train_label = getdata_undersample(X_train,y_train,aspect)
    # train_data,train_label = getdata(X_train,y_train,aspect)

    train_data = pd.DataFrame(train_data,dtype=str).fillna(' ')
    test_data,test_label = getdata(X_test,y_test,aspect)
    # test_data,test_label = getdata_undersample(X_test,y_test,aspect)

    test_data = pd.DataFrame(test_data,dtype=str).fillna(' ')

    # test_data = pd.DataFrame(test_data,dtype=str)

    # # train_data = [str (item) for item in train_data]

    # tfidf_vect_ngram = TfidfVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1,sublinear_tf=True, max_df=.5, stop_words='english', norm="l2")
    # print(train_data[0])
    # tfidf_vect_ngram.fit(train_data[0].values.astype('U'))
    # xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_data[0].values.astype('U'))
    # tfidf_vect_ngram.fit(test_data[0].values.astype('U'))
    # xvalid_tfidf_ngram = tfidf_vect_ngram.transform(test_data[0].values.astype('U'))

    # evaluation = train_model(LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2"), xtrain_tfidf_ngram, train_label, xvalid_tfidf_ngram,test_label)
    # print(evaluation)
    # ngram and vectornize of data and label

    tfidf_vect_ngram.fit(train_data[0])
    tfidf_vect_ngram.fit(test_data[0])


                # 'Bug':SGDClassifier(alpha=.001, n_iter=100, 
                #                           epsilon=0.5, loss='squared_hinge',penalty="l2", 
                #                           power_t=0.1, warm_start=False, shuffle=True, class_weight='balanced'),

    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_data[0])
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(test_data[0])

    evaluation,fulllist = train_model(clf, xtrain_tfidf_ngram, train_label, xvalid_tfidf_ngram,test_label)
    print(evaluation)
    print(fulllist)
    f1_all.append(evaluation)


     
avg_f1_score = sum(f1_all)/k
 
print('accuracy of each fold - {}'.format(f1_all))
print('Avg f1 : {}'.format(avg_f1_score))
