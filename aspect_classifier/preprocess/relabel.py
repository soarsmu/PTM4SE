import csv
import pandas as pd
input = '../RoBERTa_sentiment/data/val.csv'

final = []

with open(input, 'r')as f:
    reader = csv.reader(f)
    for read in reader:
        test = ['','']
        if read[1]=='-1':
            test[0]=read[0]
            test[1]=2
        else:
            test=read
        final.append(test)
    print(final)
    pd.DataFrame(final).to_csv('val.csv',index=False)

