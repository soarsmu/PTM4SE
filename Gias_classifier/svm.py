from datetime import date
import jieba
import numpy as np
import pandas as pd
import re
from nltk.util import ngrams
import time
import random
from gensim.models.word2vec import Word2Vec
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def getloader_undersample(datapath):

    aspect = 'Usability'
    csvfile = pd.read_csv(datapath)
    data = csvfile["sent"]
    label_text = csvfile["codes"]


    pattern = re.compile(r"([-\s.,;!?])+")
    for i in range(len(data)):

        tokens = pattern.split(data[i])
        tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
        # print(list(ngrams(tokens, 2)))
        # bi-ngram
        data[i]=list(ngrams(tokens, 2))

    print(len(data))

    label_1 = []
    label_0 = []
    data_1 = []
    data_0 = []
    count = 0
    count_1 = 0
    for i in range(len(label_text)):
        if aspect in label_text[i]:
            label_1.append(1)
            data_1.append(data[i])
            # count_1 

            count+=1
        else:
            label_0.append(0)
            data_0.append(data[i])
            count_1+=1


    print('number of label '+aspect+'in dataset: '+datapath+' is %d'%(count))
    # print('number of all data '+aspect+'in dataset: '+datapath+' is %d'%(count+count_1))

    time.sleep(1)

    # undersample
    print('length of data 0 is: %d'%(len(data_0)))
    # label_0_shuffle = np.random.shuffle(label_0)
    tartget_label = label_0[:count]
    np.random.shuffle(data_0)
    print('length of data 0 is: %d'%(len(data_0)))

    tartget_data = data_0[:count]
    tartget_data.extend(data_1)
    tartget_label.extend(label_1)

    print('number of all data '+aspect+'in dataset: '+datapath+' is %d'%(len(tartget_data)))


    return tartget_data,tartget_label

def getloader_undersample_withoutngram(datapath, aspect):



    csvfile = pd.read_csv(datapath)
    data = csvfile["sent"]
    label_text = csvfile["codes"]


    # pattern = re.compile(r"([-\s.,;!?])+")
    # for i in range(len(data)):

    #     tokens = pattern.split(data[i])
    #     tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
    #     # print(list(ngrams(tokens, 2)))
    #     # bi-ngram
    #     data[i]=list(ngrams(tokens, 2))

    # print(len(data))

    label_1 = []
    label_0 = []
    data_1 = []
    data_0 = []
    count = 0
    count_1 = 0
    for i in range(len(label_text)):
        if aspect in label_text[i]:
            label_1.append(1)
            data_1.append(data[i])
            # count_1 

            count+=1
        else:
            label_0.append(0)
            data_0.append(data[i])
            count_1+=1


    print('number of label '+aspect+'in dataset: '+datapath+' is %d'%(count))
    # print('number of all data '+aspect+'in dataset: '+datapath+' is %d'%(count+count_1))

    time.sleep(1)

    # undersample
    print('length of data 0 is: %d'%(len(data_0)))
    # label_0_shuffle = np.random.shuffle(label_0)
    tartget_label = label_0[:count]
    np.random.shuffle(data_0)
    print('length of data 0 is: %d'%(len(data_0)))

    tartget_data = data_0[:count]
    tartget_data.extend(data_1)
    tartget_label.extend(label_1)

    print('number of all data '+aspect+'in dataset: '+datapath+' is %d'%(len(tartget_data)))


    return tartget_data,tartget_label


    csvfile = pd.read_csv(datapath,sep='\t')
    data = csvfile["sent"].tolist()
    label_text = csvfile["codes"].tolist()
    label = []
    count = 0
    count_1 = 0
    for item in label_text:
        if aspect in item:
            label.append(1)
            count+=1
        else:
            label.append(0)
            count_1+=1


    return data,label

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
# fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    # return metrics.accuracy_score(predictions, valid_y)

    print(valid_y)
    print(predictions)
    return classification_report(valid_y, predictions, labels=[0,1], digits=4)

def getloader_fulllist(datapath, aspect):

    csvfile = pd.read_csv(datapath)
    data = csvfile["sent"].tolist()
    label_text = csvfile["codes"].tolist()

    print(label_text)

    label = []
    count = 0
    count_1 = 0
    for item in label_text:
        if aspect in item:
            label.append(1)
            count+=1
        else:
            label.append(0)
            count_1+=1


    return data,label


if __name__ == "__main__":

    # aspect need to be changed 
    aspect = 'Community'
    #undersample
    # train_x,train_y = getloader_undersample_withoutngram('data/train.csv', aspect)
    train_x,train_y = getloader_fulllist('data/train.csv', aspect)


    #full sample
    valid_x,valid_y = getloader_fulllist('data/val.csv', aspect)
    # valid_x,valid_y = getloader_fulllist('data/val.tsv', aspect)
    # vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1)

    # train_x = vectorizer.fit_transform(train_x)
    # valid_x = vectorizer.fit_transform(valid_x)

    # X = vectorizer.fit_transform(train_x)
    tfidf_vect_ngram = TfidfVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df = 1,sublinear_tf=True, max_df=.5, stop_words='english', norm="l2")
    tfidf_vect_ngram.fit(train_x)

                # 'Bug':SGDClassifier(alpha=.001, n_iter=100, 
                #                           epsilon=0.5, loss='squared_hinge',penalty="l2", 
                #                           power_t=0.1, warm_start=False, shuffle=True, class_weight='balanced'),

    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)

    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

    evaluation = train_model(LinearSVC(class_weight="balanced", C=1.0, loss = "hinge", max_iter=1000, penalty="l2"), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print(evaluation)
