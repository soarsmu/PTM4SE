from sklearn.model_selection import train_test_split#分割数据集
import pandas as pd
import csv
import os
import numpy as np
# def splittraindata(data,label,rate,aspect):

    # path= os.path.join('..','data_aspect')
    # # print(os.path.exists(path))
    # # exit(0)

    # os.makedirs(os.path.join(path,aspect))
    # os.makedirs(os.path.join(path,aspect,'balanced'))
    # os.makedirs(os.path.join(path,aspect,'imbalanced'))


    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=rate)

    # with open(os.path.join(path,aspect,'balanced','train.csv'),"w",encoding='utf-8',newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(X_train)):
    #         item = [X_train[i].replace('\"',''),y_train[i]]
    #         writer.writerow(item)

    # X1_train, X1_test, y1_train, y1_test = train_test_split(X_test, y_test, test_size=0.5)   
    # with open('test.csv',"w",encoding='utf-8',newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(y1_train)):
    #         item = [X1_train[i].replace('\"',''),y1_train[i]]
    #         writer.writerow(item)
    # with open('val.csv',"w",encoding='utf-8',newline='') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(X1_test)):
    #         item = [X1_test[i].replace('\"',''),y1_test[i]]
    #         writer.writerow(item)



# def csvtodata():
#     pf = pd.read_csv('../data/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv')
#     label = pf['codes'].values.tolist()
#     data = pf['sent'].values.tolist()
#     return label,data

# 把csv切割成训练集和测试集
def csv_to_train_test_val(filepath,test_ratio=0.1):
    '''
    input csv
    output pandaframe
    '''
    df = pd.read_csv(filepath, encoding='utf-8')
    df = df.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(test_ratio * df.shape[0]/2))
    df_test, df_val, df_train = df.iloc[:cut_idx],df.iloc[cut_idx:2*cut_idx], df.iloc[2*cut_idx:]
    print(df_test)
    print(df_val)
    print(df_train)

    return df_train,df_test,df_val

    #  df_test.shape, df_train.shape  # (3184, 12) (318, 12) (2866, 12)

# 从csv中提取数据
# def extract_aspect_from_csv(filepath,aspect):
#     df = pd.read_csv(filepath, encoding='utf-8')
#     data = np.array(df)
#     # print(data)
#     count=0
#     target_csv = []
#     other_csv = []
#     for item in data:
#         # print(item[4])
#         if aspect in item[4]:
#             # count+=1
#             # print(item[4])
#             target_csv.append(item)
#         else:
#             other_csv.append(item)
#     print(other_csv)
#     target = pd.DataFrame(data=target_csv, index=["thread", "tid",'sent','ManualLabel','codes','stakeholder','signal','intent','action','LEN','LEN'])
#     target.to_csv()
# label,data =csvtodata()
# splittraindata(data,label,0.2,aspect="Usability")
# extract_aspect_from_csv('../data/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv',aspect = 'Usability')
df_train,df_test,df_val = csv_to_train_test_val('../data/BenchmarkUddinSO-ConsoliatedAspectSentiment.csv',0.2)
df_train.to_csv('../data_aspect/train.tsv',sep='\t')
df_test.to_csv('../data_aspect/test.tsv',sep='\t')
df_val.to_csv('../data_aspect/val.tsv',sep='\t')