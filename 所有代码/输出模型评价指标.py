import pandas as pd
import numpy as np
from parameter import *
import warnings
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_score, recall_score, f1_score

#计算xgb模型的结果情况
# raw_data_2_0=pd.read_csv(r'C:\Users\ljs11\Desktop\实验设计作业\Final_code\data\Result\XGB原始_结果.csv',header=None,encoding='utf8',names = ['id','flag_raw'])
#计算改进xgb模型的结果情况
# raw_data_2_0=pd.read_csv(r'C:\Users\ljs11\Desktop\实验设计作业\Final_code\data\Result\XGB改进_结果.csv',header=None,encoding='utf8',names = ['id','flag_raw'])
#计算lgb模型的结果情况
# raw_data_2_0=pd.read_csv(r'C:\Users\ljs11\Desktop\实验设计作业\Final_code\data\Result\LGB改进_结果.csv',header=None,encoding='utf8',names = ['id','flag_raw'])
#计算MLPClassifier神经网络模型的结果情况
# raw_data_2_0=pd.read_csv(r'C:\Users\ljs11\Desktop\实验设计作业\Final_code\data\Result\神经网络_结果.csv',header=None,encoding='utf8',names = ['id','flag_raw'])
#计算RandomForestClassifier神经网络模型的结果情况
raw_data_2_0=pd.read_csv(r'C:\Users\ljs11\Desktop\实验设计作业\Final_code\data\Result\随机森林_结果(30).csv',header=None,encoding='utf8',names = ['id','flag_raw'])


#设定AUC阈值
sill=np.arange(0,1,0.01)
for i in sill:
# sill=0.5

#根据阈值判定是否会违约
    negative_0 = raw_data_2_0[raw_data_2_0['flag_raw']<i]
    positive_1 = raw_data_2_0[raw_data_2_0['flag_raw']>=i]
    negative_0['flag_pred'] = 0
    positive_1['flag_pred'] = 1
    pieces = {'0': negative_0, '1': positive_1}
    new_data_2_0=pd.concat(pieces)

    #根据选取的预测集寻找原始数据
    data = pd.read_csv(PROCESSING_DATA)
    data = data[data['isTest'] == -1]
    update_data = pd.merge(data, new_data_2_0, how='right', on='id')

    # 计算并输出 precision & recall & f1-score & support参数
    # print(classification_report(update_data['flag'], update_data['flag_pred']))

    #计算AUC所需参数
    fpr, tpr, thresholds = metrics.roc_curve(update_data['flag'], update_data['flag_pred'], pos_label=1)

    warnings.filterwarnings('ignore')
    #绘制AUC曲线并输出AUC
    # plt.plot(fpr,tpr,marker = 'o')
    # plt.show()
    AUC = auc(fpr, tpr)
    # print("sill=",i,"AUC =",AUC)
    # #输出精确率 （提取出的正确信息条数 / 提取出的信息条数）
    # print("sill=",i,'Precision: %.3f' % precision_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']))
    # #输出召回率 （提出出的正确信息条数 / 样本中的信息条数）
    # print("sill=",i,'Recall: %.3f' % recall_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']))
    # #输出F1-score  （正确率*召回率*2 /（正确率+召回率））
    # print("sill=",i,'F1: %.3f' % f1_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']))

    print("sill=", i,'/'
          'AUC=', AUC,'/'
          'Precision=%.3f' %precision_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']),'/'
          'Recall=%.3f' %recall_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']),'/'
          'F1=%.3f' %f1_score(y_true=update_data['flag'], y_pred=update_data['flag_pred']))
