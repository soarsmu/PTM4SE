from data.dataloader import get_train_valid_Bert_undersample
from utils.training_utils import save_checkpoint,save_metrics
import torch.nn as nn
import os 
from utils.training_utils import load_checkpoint
from sklearn.metrics import classification_report, confusion_matrix
from transformers import XLNetForSequenceClassification, XLNetTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import random
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from pylab import rcParams
from sklearn.model_selection import StratifiedKFold

from torch import nn, optim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
      model.zero_grad()
      model = model.train()
      losses = []
      acc = 0
      counter = 0
  
      for d in data_loader:
            input_ids = d["input_ids"].reshape(16,160).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
            loss = outputs[0]
            logits = outputs[1]                  # loss, _ = output

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            try:      
                  accuracy = metrics.accuracy_score(targets.tolist(), prediction.tolist())
            except:
                  print(targets)
                  print(prediction)
            acc += accuracy
            losses.append(loss.item())
            
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            counter = counter + 1
      return acc / counter, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
      model = model.eval()
      losses = []
      f1_sum = 0
      counter = 0
      y_pred = []
      y_true = []
      with torch.no_grad():
            for d in data_loader:
                  input_ids = d["input_ids"].reshape(16,160).to(device)
                  attention_mask = d["attention_mask"].to(device)
                  targets = d["targets"].to(device)
                  
                  outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
                  loss = outputs[0]
                  logits = outputs[1]

                  _, prediction = torch.max(outputs[1], dim=1)
                  targets = targets.cpu().detach().numpy()
                  prediction = prediction.cpu().detach().numpy()
                  f1 = metrics.f1_score(targets, prediction)

                  f1_sum += f1
                  y_pred.extend(prediction)
                  y_true.extend(targets.tolist())
                  losses.append(loss.item())
                  counter += 1

      return metrics.f1_score(y_true, y_pred,average='weighted'), np.mean(losses),classification_report(y_true, y_pred, labels=[0,1], digits=4),metrics.precision_score(y_true, y_pred,average='weighted'),metrics.recall_score(y_true, y_pred,average='weighted')


def writeoutput(info):
      mylog = open('record_tmp_2.log',mode = 'a', encoding='utf_8')
      print(info,file=mylog)
      mylog.close()

def translate(label,aspect):

      label_all = []
      count = 0

      for i in range(len(label)):
            if aspect in label[i]:
                  label_all.append(1)
                  count+=1
            else:
                  label_all.append(0)
      return label_all,count
#分层抽样   x1是数据的标号，x1里拆分出10%的验证集，ydigit是所有数据的label 
def returnlabel(x1,y_digit,data):
      count=0
      count_negative = 0
      y_pos = []
      y_neg = []
      for x in x1:
            if y_digit[x]==1:
                  count+=1
                  y_pos.append(x)
            else:
                  count_negative+=1
                  y_neg.append(x)
      # ratio = float(count/(count+count_negative))
      # 1/10正例的个数，四舍五入
      pos_train,pos_val = train_test_split(y_pos,train_size=0.9)
      neg_train,neg_val = train_test_split(y_neg,train_size=0.9)


      return pos_train+neg_train,pos_val+neg_val

if __name__ == "__main__":

      PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

      #data part
      datapath = '/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv'
      data = pd.read_csv(datapath,sep=',')
      X = data['sent'] # data
      y = data['codes'] # label 

      # 10 folder cross validation
      k = 10
      kf = StratifiedKFold(n_splits=k, random_state=None)

      aspects = ['Security','Usability','Security','Community','Compatibility','Portability','Documentation','Bug','Legal','OnlySentiment','Other']
      # aspects = ['Usability','Security','Community','Compatibility','Portability','Documentation','Bug','Legal','OnlySentiment','Other']
      # aspects = ['Usability','Security','Community','Compatibility']
      # aspects = ['Portability','Documentation','Bug','Legal','OnlySentiment','Other']

      # aspects = ['Performance','Usability']
      EPOCHS = 6
      BATCH_SIZE = 32
      # MAX_LEN_LIST = [200,160]
      learn_rate_list = [5e-5,3e-5,1e-5,8e-6]
      undersample = [True,False]
      # learn_rate_list = [1e-5,3e-5]
      # EPOCHS = 1
      # MAX_LEN_LIST = [200,100]
      # learn_rate_list = [1e-5]


      for aspect in aspects:
            # 保存数据
            dataframe = pd.DataFrame(columns=('aspect','learn_rate','sample strategy','best_precision','best recall','best F-1'))
      
            count = 0
            # 对每个aspect做一个十折, 这里是分层抽样
            y_digit,true_label_number = translate(y,aspect)
            # 十折
            for x1,x2 in kf.split(X,y_digit):
                  # 分别计算 full sample 和 undersample
                  # for samplestrategy in undersample:

                  #分层抽样实现训练集和测试集,经测试无问题
                  train_stratified,val_stratified = returnlabel(x1,y_digit,X)

                  # sample strategy
                  for samplestrategy in undersample:

                        best_F1 = -1
                  
                        train_iter, val_iter, test_iter = get_train_valid_Bert_undersample(samplestrategy,X,y_digit,aspect,train_stratified,val_stratified,x2,tokenizer, max_seq_lenght=160)

                        for learn_rate in learn_rate_list:

                              model = XLNetForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2)
                              model = model.to(device)

                              param_optimizer = list(model.named_parameters())
                              no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                              optimizer_grouped_parameters = [
                                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
                                                            ]

                              best_epoch_F1 = -1

                              optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate)

                              total_steps = len(train_iter) * EPOCHS

                              scheduler = get_linear_schedule_with_warmup(
                              optimizer,
                              num_warmup_steps=0,
                              num_training_steps=total_steps
                              )
                              
                              # 输出best epoch loss
                              for epoch in range(EPOCHS):

                                    
                                    print(f'Epoch {epoch + 1}/{EPOCHS}')
                                    print('-' * 10)

                                    train_acc, train_loss = train_epoch(
                                          model,
                                          train_iter,     
                                          optimizer, 
                                          device, 
                                          scheduler, 
                                          len(train_iter)
                                    )

                                    print(f'Train loss {train_loss} Train accuracy {train_acc}')

                                    val_F_1, val_loss, metrics_all, val_precision, val_recall = eval_model(
                                          model,
                                          val_iter, 
                                          device, 
                                          len(val_iter)
                                    )

                                    print(f'Val loss {val_loss} Val F-1 {val_F_1}')
                                    print()

                                    # 这里每个epoch保存一个最好的模型，然后依次所有的模型进行对比，最后输出最好的
                                    if val_F_1 > best_epoch_F1:
                                          best_epoch_F1 = val_F_1
                                          print('the val F-1 is surpass the best F-1 on this fold, model has been saved \n')

                                          path = '/workspace/RoBERTa_textclassification/aspect_classifier/XLNet_classification/model/'+aspect+str(count)+'Roberta_model.bin'
                                          torch.save(model.state_dict(), path)

                              model.load_state_dict(torch.load(path))
                              test_F_1, test_loss, metrics_all, test_precision, test_recall = eval_model(
                                          model,
                                          test_iter, 
                                          device, 
                                          len(test_iter)
                                    )
                              if test_F_1 > best_F1:
                                    best_F1 = test_F_1
                                    best_metrics = metrics_all
                                    best_precision = test_precision
                                    best_recall = test_recall
                                    best_learn_rate = learn_rate
                                    torch.save(model.state_dict(), path)
                              del model

                        record_best = pd.DataFrame([[aspect,best_learn_rate,samplestrategy,best_precision,best_recall,best_F1]])
                        # {'aspect':[aspect],'learn_rate':[best_learn_rate],'sample strategy:':[samplestrategy],'best_precision':[best_precision],'best recall':[best_recall],'best F-1':[best_F1]}

                        dataframe=dataframe.append(record_best,ignore_index=True) 
                        print(record_best)
                        print(best_metrics)
                        csvfilename=str(aspect)+' result.csv'
                        dataframe.to_csv(csvfilename,mode='a',sep=',')
                        dataframe=dataframe.drop(index=dataframe.index)
                  count += 1

