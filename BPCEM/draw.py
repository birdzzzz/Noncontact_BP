#encoding: utf-8
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import os
import numpy as np
import matplotlib.lines as mlines
def loaddata1(path):
    labelarr = []
    date = os.listdir(path)
    for i in range(len(date)):
        datafolder = os.path.join(path, date[i])
        vonlunteers = os.listdir(datafolder)
        for j in range(len(vonlunteers)):
            labelpath = datafolder + '/' + vonlunteers[j] + '/' + 'bp.csv'
            labelrecording = loadlabel(labelpath)
            labelarr.append(labelrecording)
    return labelarr
def loadlabel(labelpath):
    recording = []
    with open(labelpath, 'r', encoding='UTF-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            recording.append(float(row[0]))
            recording.append(float(row[1]))
    recording = np.array(recording)
    return recording

dd=['test','val','train']
arr=[]
for d in dd:
    path = "/home/som/lab-data/jmrnewdata/laboratorydata/"+d+"/"
    a=loaddata1((path))
    arr.extend(a)
arr = np.array(arr)
arr0=arr[:,0]
arr1=arr[:,1]

plt.figure('finalalex')
ax = plt.gca()
# 设置x轴、y轴名称
ax.set_xlabel('$SBP_{gt}$', fontproperties='Times New Roman', fontsize=13)
ax.set_ylabel('$DBP_{gt}$', fontproperties='Times New Roman', fontsize=13)

# s为点的大小
ax.scatter(arr0, arr1, c='b', s=30, alpha=0.5)

plt.yticks(fontproperties='Times New Roman', size=13)
plt.xticks(fontproperties='Times New Roman', size=13)

plt.grid(True)
plt.grid(color='r', linestyle='--', linewidth=0.2)  # 修改网格颜色，类型为虚线

plt.show()





# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False
# data = pd.read_csv('/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/BP1.csv')
# data2 = pd.read_csv('/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/BP2.csv')
# data3 = pd.read_csv('/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/BP3.csv')
# data4 = pd.read_csv('/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/BP4.csv')
# xdata = []
# xdata2 =[]
# xdata3 =[]
# xdata4=[]
# ydata = []
# ydata2 = []
# ydata3 = []
# ydata4 = []
# xdata = data.ix[:,'epoch']   #将csv中列名为“列名1”的列存入xdata数组中
# 			#如果ix报错请将其改为loc
# xdata2 = data.ix[:,'epoch']
# xdata3 = data.ix[:,'epoch']
# xdata4 = data.ix[:,'epoch']
# ydata = data.ix[:,'val_mean_absolute_error']   #将csv中列名为“列名2”的列存入ydata数组中
# ydata2 = data2.ix[:,'val_mean_absolute_error']
# ydata3 = data3.ix[:,'val_mean_absolute_error']#val_mean_absolute_error
# ydata4 = data4.ix[:,'val_mean_absolute_error']
# # plt.plot(xdata,ydata,'y-',label=u'hand1',linewidth=1)
# # plt.plot(xdata,ydata2,'b-',label=u'hand2',linewidth=1)
# # plt.plot(xdata,ydata3,'g-',label=u'hand3',linewidth=1)
# plt.plot(xdata,ydata4,'r-',label=u'hand4',linewidth=1)
# plt.title(u"result",size=10)   #设置表名为“表名”
# plt.legend()
# plt.xlabel(u'epoch',size=10)   #设置x轴名为“x轴名”
# plt.ylabel(u'valmae',size=10)   #设置y轴名为“y轴名”
# plt.xlim(0,100)
# plt.show()

# r"/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/mySEfft.csv"