# Introduction

APIs   (Application   Programming   Interfaces)   arereusable  software  libraries  and  are  building  blocks  for  modernrapid  software  development.  Previous  research  shows  that  pro-grammers  frequently  share  and  search  for  reviews  of  APIs  onthe mainstream software question and answer (Q&A) platformslike Stack Overflow, which motivates researchers to design tasksand  approaches  related  to  process  API  reviews  automatically.Among these tasks, classifying API reviews into different aspects(e.g.,  performance  or  security),  which  is  called  theaspect-basedAPI  review  classification,  is  of  great  importance.  The  currentstate-of-the-art  (SOTA)  solution  to  this  task  is  based  on  thetraditional  machine  learning  algorithm.  Inspired  by  the  greatsuccess  achieved  by  pre-trained  models  on  many  software  engi-neering tasks, this study fine-tunes six pre-trained models for theaspect-based  API  review  classification  task  and  compares  themwith  the  current  SOTA  solution  on  an  API  review  benchmarkcollected  by  Uddin  et  al.  The  investigated  models  include  fourmodels  (BERT,  RoBERTa,  ALBERT  and  XLNet)  that  are  pre-trained on natural languages, BERTOverflow that is pre-trainedon  text  corpus  extracted  from  posts  on  Stack  Overflow,  andCosSensBERT  that  is  designed  for  handling  imbalanced  data.The  results  show  that  all  the  six  fine-tuned  models  outperformthe traditional machine learning-based tool. More specifically, theimprovement  on  the  F1-score  ranges  from21.0%to30.2%.  Wealso find that BERTOverflow, a model pre-trained on the corpusfrom  Stack  Overflow,  does  not  show  better  performance  thanBERT. The result also suggests that CosSensBERT also does notexhibit  better  performance  than  BERT  in  terms  of  F1,  but  it  isstill worthy of being considered as it achieves better performanceon  MCC  and  AUC.

# Dataset
API review dataset have been used. The sources of the dataset are noted in the paper. Credit to the original authors. You can download the original dataset in the following sources.
+ [API review dataset](https://github.com/giasuddin/OpinionValueTSE):



# Approaches
## PTMs
## baseline

# Scripts


PTM-API-AspectClassifier
