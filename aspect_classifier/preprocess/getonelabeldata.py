import csv
import os
# from setting import *
import pandas as pd
import xlrd
import csv
def readAPIname(api):
    return api.split('.')[-1]

def removestopwordsfromAPI(api):
    stoplist = ['test', 'sample', 'example', 'demo', 'code']
    tmp = api
    for stopword in stoplist:
        tmp = tmp.replace(stopword+'.','')
    return tmp


def createfuzzyapi(api):
    api = removestopwordsfromAPI(api)
    globalname = api.split('.')[0]
    provider = api.split('.')[1]
    simple = readAPIname(api)
    combi1 = str(provider+'.'+simple)
    combi2 = str(provider+' '+simple)
    combi3 = str(simple+' '+provider)
    combi2 = str(provider+'-'+simple)
    return [combi1,combi2,combi3]


def csvtodic(path):
    print(path)
    output = []
    api = ''
    with open(path)as f:
        readers = csv.reader(f)
        modules =[]
        resourcelink = []
        homepage = []
        for read in readers:
            if api=='':
                api = read[0]
            if read[0]==api:
                modules.append(read[1])
                resourcelink.append(read[2])
                homepage.append(read[4])
            if read[0]!=api:
                # print(modules)
                tmp={}
                tmp["api"]=api
                tmp['module']=modules
                tmp['resourcelink']=resourcelink
                tmp['homepage']=homepage
                if tmp['api']!='API name':
                    output.append(tmp)
                modules= []
                resourcelink = []
                homepage = []
                api=read[0]
        print(modules)    
        print(output)

# class api:
#     #api的基类

#     def __init__(self,fullname,modules,mainpage,mavenpage):
#         self.fullname = fullname
#         self.modules= modules
#         self.mainpage = mainpage
#         self.mavenpage = mavenpage

def getsentimentdata():
    input = '../data/BenchmarkUddinSO-ConsoliatedAspectSentiment.xls'
    xlsx = xlrd.open_workbook(input)
    sheet1 = xlsx.sheets()[0]
    sheet1_nrows = sheet1.nrows
    data = []
    label = []
    for i in range(sheet1_nrows):
        if i ==0:
            continue
        data.append(sheet1.row_values(i)[2])
        label.append(int(float(sheet1.row_values(i)[3])))
        # if 'p' in str(sheet1.row_values(i)[3]):
        #     label.append(1)
        # if 'n' in str(sheet1.row_values(i)[3]):
        #     label.append(-1)
        # if str(sheet1.row_values(i)[3])=='o':
        #     label.append(0)
        # else:
        #     print("false")
        #     print(sheet1.row_values(i)[3])
    label_new = []

    dic_pd = {}
    dic_pd["label"]=label
    dic_pd["data"]=data
    return dic_pd

def dic2csv(dic):
    import pandas as pd
    data_df = pd.DataFrame(dic)
    data_df.to_csv('../data/data_df.csv')


def savedata(aspect):
    input = '../data/aspect_gias_benchmark.csv'
    count = []
    outputdir = "/workspace/pipeline/aspect_classifier/data/"
    count = []
    with open(input)as f:
        reader = csv.reader(f)
        for read in reader:
            # print(read[1])
            if aspect in read[1]:
                count.append([read[0].replace('\"','').replace('\\','').replace('\'',''),read[1].replace('\"','').replace('\\','').replace('\'','')])
                # print(read[1])
    name = ['data','label']
    tmp = pd.DataFrame(columns=name,data=count)
    print(tmp)
    tmp.to_csv(os.path.join(outputdir,str(aspect+".csv")),index=False)    


if __name__=='__main__':
#     input = 'APIinfo_3first.csv'
#     # util.xlsvtocsv(filename, outputname)
#     apiinfo = []
#     with open(os.path.join(work_dir,'API_initial_three.csv'))as f:
#         reader = csv.reader(f)
#         for read in reader:
#             # read[0] = api(read[0],read[1],read[2],read[3])
#             apiinfo.append(read)
#     token = []
#     for case in apiinfo:
#         tmp = []
#         tmp.append(readAPIname(case[0]))
#         tmp.append(case[0])
#         tmp.append(case[-1])
#         tmp.append(case[-2])
#         tmp.extend(createfuzzyapi(case[0]))
#         print(tmp)
#     # token 用来匹配

    # get aspect based data
    input = 'aspect_gias_benchmark.csv'
    # util.xlsvtocsv(filename, outputname)
    apiinfo = []
    savedata(aspect="Security")
    savedata(aspect="Others")
    savedata(aspect="Performance")
    savedata(aspect="Community")
    savedata(aspect="Compatibility")
    savedata(aspect="Bug")
    savedata(aspect="Usability")
    savedata(aspect="Documentation")
    savedata(aspect="OnlySentiment")
    savedata(aspect="Legal")
    savedata(aspect="Portability")

    # get sentiment data
    # getsentimentdata()
    # dic2csv(getsentimentdata())
    # # readcsv()
    # df=pd.read_csv('../data/data_df.csv')
    # print(df)