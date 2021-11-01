import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
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

def getloader(X,y_digit,train_stratified):

    pos_train = []
    neg_train = []  
    for data in train_stratified:
        if y_digit[data]==1:
            pos_train.append(X[data])
        else:
            neg_train.append(X[data])
    
    tartget_data = pos_train+neg_train
    tartget_label = [1]*len(pos_train)+[0]*len(neg_train)

    return tartget_data,tartget_label

datapath = '/workspace/RoBERTa_textclassification/aspect_classifier/costsensitivebert/Cost-Sensitive_Bert_and_Transformers/examples/glue_data/CoLA/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv'
data = pd.read_csv(datapath,sep=',')
X = data['sent'] # data
y = data['codes'] # label 

k = 10
kf = StratifiedKFold(n_splits=k, random_state=None)


# 十折
aspects = ['Security','Usability','Performance','Community','Compatibility','Portability','Documentation','Bug','Legal','OnlySentiment','Other']
for aspect in aspects:
    y_digit,true_label_number = translate(y,aspect)
    count = 0
    for x1,x2 in kf.split(X,y_digit):
        train_tartget_data,train_tartget_label = getloader(X,y_digit,x1)
        val_tartget_data,val_tartget_label = getloader(X,y_digit,x2)

        train_iter = {
            'data':train_tartget_data,
            'label':train_tartget_label
        }
        train_csv = pd.DataFrame(train_iter)

        val_iter = {
            'data':val_tartget_data,
            'label':val_tartget_label
        }
        val_csv = pd.DataFrame(val_iter)

        # make file
        tmp_path = '../glue_data/CoLA/'+aspect+'/'+str(count)+'/'
        # print('../glue_data/CoLA/'+aspect+'/'+str(count)+'/')
        if not os.path.exists('../glue_data/CoLA/'+aspect+'/'):
            os.mkdir('../glue_data/CoLA/'+aspect+'/')
        if not os.path.exists('../glue_data/CoLA/'+aspect+'/'+str(count)+'/'):
            os.mkdir(tmp_path)
        train_path = tmp_path+'train.tsv'
        train_csv.to_csv(train_path,sep='\t')
        val_path = tmp_path+'val.tsv'
        val_csv.to_csv(val_path,sep='\t')

        count+=1