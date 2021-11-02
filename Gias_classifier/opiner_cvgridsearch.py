#Importing required librariesimport pandas as pd
from os import pread
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
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.metrics import classification_report,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler as  ros


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


def getdata_oversample(data,label,aspect):
    # make the label with target aspect as 1, without as 0
    # randomly select same number data from dataset0 equals to dataset0

    label_all = []
    data_label1 = []
    data_label0 = []

    ros_1 = ros(sampling_strategy='minority',random_state=0)

    # divided positive samples and negative samples
    for i in range(len(label)):
        if aspect in label[i]:
            data_label1.append(data[i])
        else:
            data_label0.append(data[i])

    # randomly select data points from negative samples, number equals to length of positive samples
    # if len(data_label1)<=len(data_label0):
    extracted_data0 = data_label0
    # else:
    #     # not realistic
    #     extracted_label0 = random.sample(data_label0,len(data_label1))

    # make the label list
    label0 = [0]*len(extracted_data0)
    extracted_data0.extend(data_label1)

    label1 = [1]*len(data_label1)
    label0.extend(label1)
    extracted_data0 = pd.DataFrame(extracted_data0)
    # print(extracted_data0)
    # print(label0)
    # exit(0)
    X_resample,y_resample = ros_1.fit_resample(extracted_data0[0].values.reshape(-1,1),label0)
    # print(X_resample)
    # print(y_resample)
    # exit(0)
    return X_resample,y_resample


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

    return f1_score(valid_y, predictions,average='weighted')\
        ,matthews_corrcoef(valid_y, predictions)\
        ,classification_report(valid_y, predictions, labels=[0,1], digits=4)\
        ,f1_score(valid_y, predictions,average='binary')
        


#Loading the dataset
datapath = '/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv'
data = pd.read_csv(datapath,sep=',')
X = data['sent'] # data
y = data['codes'] # label 

#Implementing cross validation, shuffle optional
k = 10
kf = StratifiedKFold(n_splits=k, shuffle=True,random_state=True)
# kf = StratifiedKFold(n_splits=k, random_state=True,shuffle=True)
pos_f1 = []
f1_all = []
mcc_all = []
recall_all = []
auc_all = []
pre_all = []
# parameters

# if aspect == 'Other':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True, norm="l2")
#     clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
# if aspect == 'Usability':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l2")
#     countvect = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)
#     clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
# if aspect == 'Performance':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True, norm="l2")
#     clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
#     countvect = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)
# if aspect == 'Security':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True)
#     clf =SGDClassifier(alpha=.0001, max_iter=2000, 
#                                           epsilon=0.5, loss='log',penalty="l2", 
#                                           power_t=0.5, warm_start=False, shuffle=True, class_weight='balanced')
#     countvect = CountVectorizer(stop_words = 'english')
# if aspect == 'Community':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l1")
#     # clf = LinearSVC(class_weight="balanced", C=10.0, loss = "squared_hinge", max_iter=1000, penalty="l2")
#     clf =SGDClassifier(alpha=.0001, max_iter=2000, 
#                                           epsilon=0.5, loss='log',penalty="l2", 
#                                           power_t=0.5, warm_start=False, shuffle=True, class_weight='balanced')
#     countvect =CountVectorizer(stop_words = 'english')                                          
# if aspect == 'Bug':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True, norm="l2")
#     clf = SGDClassifier(alpha=.001, max_iter=100, 
#                                           epsilon=0.5, loss='squared_hinge',penalty="l2", 
#                                           power_t=0.1, warm_start=False, shuffle=True, class_weight='balanced')
#     countvect = CountVectorizer(stop_words = 'english')                                        
# if aspect == 'Compatibility':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l1")
#     clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
#     countvect = CountVectorizer(ngram_range = (1,3), token_pattern = r'\b\w+\b', min_df = 1)
# if aspect == 'Documentation':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l1")
#     clf = LinearSVC(class_weight="balanced", C=20.0, loss = "squared_hinge", max_iter=1000, penalty="l2")
#     countvect = CountVectorizer(ngram_range = (1,3), token_pattern = r'\b\w+\b', min_df = 1)

# if aspect == 'Legal':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l1")
#     clf = SGDClassifier(alpha=.0001, epsilon=0.5, loss='hinge', 
#                                   max_iter=50,penalty="l2", power_t=0.5, shuffle=True,warm_start=False, class_weight='balanced')
#     countvect = CountVectorizer(stop_words = 'english')
# if aspect == 'Portability':
#     tfidf_vect_ngram =  TfidfTransformer(sublinear_tf=True,norm="l2")
#     clf = LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2")
#     countvect = CountVectorizer(stop_words = 'english')

aspects = ['Security','Usability','Performance','Community','Compatibility','Portability','Documentation','Bug','Legal','OnlySentiment','Other']

# training process with 10 folder cross validation
aspect = 'Bug'
for train_index , test_index in kf.split(X,y):
    
    # get the index of data and label based on aspects
    X_train , X_test = X[train_index].tolist(),X[test_index].tolist()
    y_train , y_test = y[train_index].tolist(), y[test_index].tolist()
    # define if the data is oversampled
    train_data,train_label = getdata_undersample(X_train,y_train,aspect)
    # train_data,train_label = getdata(X_train,y_train,aspect)
    # train_data,train_label = getdata_oversample(X_train,y_train,aspect)

    train_data = pd.DataFrame(train_data,dtype=str).fillna(' ')
    # test_data,test_label = getdata(X_test,y_test,aspect)
    test_data,test_label = getdata_undersample(X_test,y_test,aspect)

    test_data = pd.DataFrame(test_data,dtype=str).fillna(' ')

    data = train_data[0].values.tolist()
    test_data= test_data[0].values.tolist()

    text_clf_1 = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer(sublinear_tf=True)),
                        ('clf', SGDClassifier(max_iter=3000,class_weight='balanced')),])
    # text_clf = text_clf.fit(data, train_label)
    from sklearn.model_selection import GridSearchCV
    parameters_1 = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3),
                'tfidf__norm':('l1','l2')

                    }

    text_clf_2 = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer(sublinear_tf=True)),
                        ('clf', LinearSVC(max_iter=3000,class_weight='balanced')),])

    
    parameters_2 = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
                'tfidf__use_idf': (True, False),
                'clf__C':(1e-4, 1e-3,1e-2, 1e-1,1.0,10.0,20.0),
                'clf__loss': ('squared_hinge','hinge'),
                'tfidf__norm':('l1','l2')
                    }
    # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,scoring='f1_weighted')
    gs_clf_1 = GridSearchCV(text_clf_1, parameters_1, n_jobs=-1,scoring='f1_weighted')
    gs_clf_2 = GridSearchCV(text_clf_2, parameters_2, n_jobs=-1,scoring='f1_weighted')
# f1_weighted

    gs_clf_1 = gs_clf_1.fit(data, train_label)
    gs_clf_2 = gs_clf_2.fit(data, train_label)



    if gs_clf_1.best_score_ >gs_clf_2.best_score_:

        val_y  = gs_clf_1.best_estimator_.predict(test_data)

    else:
        val_y  = gs_clf_2.predict(test_data)
    
    f_1 = f1_score(test_label,val_y,average='weighted')
    mcc =matthews_corrcoef(test_label,val_y)
    recall = recall_score(test_label, val_y,average='weighted')
    auc = roc_auc_score(test_label, val_y,average='weighted')
    precision = precision_score(test_label, val_y,average='weighted')
    f_1_final = f_1.tolist()
    #开始写这个estimator的验证
    f1_all.append(f_1)
    mcc_all.append(mcc)
    recall_all.append(recall)
    pre_all.append(precision)
    auc_all.append(auc)
#     pos_f1.append(f_1_pos)
    
avg_f1_score = sum(f1_all)/k
avg_mcc_score = sum(mcc_all)/k
avg_pre_score = sum(pre_all)/k
avg_recall_score = sum(recall_all)/k
avg_auc_score = sum(auc_all)/k
# avg_posf1_score = sum(pos_f1)/k
# print('accuracy of each fold - {}'.format(f1_all))
print('aspect is : %s'%(aspect))
print('Avg pre : {}'.format(avg_pre_score))
print('Avg recall : {}'.format(avg_recall_score))
print('Avg f1 : {}'.format(avg_f1_score))
print('Avg mcc : {}'.format(avg_mcc_score))
print('Avg auc : {}'.format(avg_auc_score))
# print('Avg posf1 : {}'.format(avg_posf1_score))
