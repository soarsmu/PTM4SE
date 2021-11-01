import random
from typing import Tuple
import pandas as pd
from transformers import RobertaTokenizer,BertTokenizer
import os
from data.dataset_class import ImdbDataset
from torch.utils.data.dataloader import DataLoader
import time
import numpy as np


def getloader_undersample(X,y_digit,train_stratified,aspect,tokenizer,max_seq_lenght,max_size,drop_last=True):
    # aspect = 'Other'
    # for i in range(len(train_stratified)):
    pos_train = []
    neg_train = []
    pos_test = []
    neg_test = []   
    for data in train_stratified:
        if y_digit[data]==1:
            pos_train.append(X[data])
        else:
            neg_train.append(X[data])
    
    neg_train = random.sample(neg_train,len(pos_train))


    tartget_data = pos_train+neg_train
    tartget_label = [1]*len(pos_train)+[0]*len(neg_train)


    dataset = ImdbDataset(tartget_data,tartget_label,tokenizer,max_len=max_size)
    return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)

def getloader(X,y_digit,train_stratified,aspect,tokenizer,max_seq_lenght,max_size,drop_last=True):
    # aspect = 'Other'
    # for i in range(len(train_stratified)):
    pos_train = []
    neg_train = []
    pos_test = []
    neg_test = []   
    for data in train_stratified:
        if y_digit[data]==1:
            pos_train.append(X[data])
        else:
            neg_train.append(X[data])
    
    tartget_data = pos_train+neg_train
    tartget_label = [1]*len(pos_train)+[0]*len(neg_train)


    dataset = ImdbDataset(tartget_data,tartget_label,tokenizer,max_len=max_size)
    return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)


# def getloader(aspect,datapath,tokenizer,max_seq_lenght,max_size,drop_last=True):
#     # aspect = 'Other'
#     csvfile = pd.read_csv(datapath)
#     data = csvfile["sent"].tolist()
#     label_text = csvfile["codes"].tolist()
#     label = []
#     count = 0
#     count_1 = 0
#     for item in label_text:
#         if aspect in item:
#             label.append(1)
#             count+=1
#         else:
#             label.append(0)
#             count_1+=1

#     print('number of label '+aspect+'in dataset: '+datapath+' is %d'%(count))
#     print('number of all data '+aspect+'in dataset: '+datapath+' is %d'%(count+count_1))


#     time.sleep(1)

#     dataset = ImdbDataset(data,label,tokenizer,max_len=max_size)
#     # print(dataset.__dict__)
#     return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)

def get_train_valid_Bert_undersample(samplestrategy,X,y_digit,aspect,train_stratified,val_stratified,x2, tokenizer, max_seq_lenght):
    # tokenizer = RobertaTokenizer.from_pretrained(bert_model)
    if samplestrategy == True:
        loader_train = getloader_undersample(X,y_digit,train_stratified,aspect,tokenizer,max_seq_lenght,max_size=max_seq_lenght)
    else:
        loader_train = getloader(X,y_digit,train_stratified,aspect,tokenizer,max_seq_lenght,max_size=max_seq_lenght)

    loader_test = getloader(X,y_digit,x2,aspect,tokenizer,max_seq_lenght,max_size=max_seq_lenght)

    loader_valid = getloader(X,y_digit,val_stratified,aspect,tokenizer,max_seq_lenght,max_size=max_seq_lenght)
    return loader_train, loader_valid, loader_test


