
from keras.models import *
from utils import lr_schedule
import  numpy as np
import time, sys, queue, random
from multiprocessing.managers import BaseManager
import cv2
from keras import backend as K
from keras.layers import *
import os
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras import metrics,losses
from keras.utils import multi_gpu_model
import tensorflow as tf
import csv
from time import *
time00=time()
loss = losses.mean_squared_error
modelpath='/home/som/lab/jmr/CBAM-keras-master/CBAM-keras-master/saved_models/columnBP715.hdf5'##########################################################################################################
bpmodel=load_model(modelpath,compile=False)#,compile=false
bpmodel.summary()
bpmodel.compile(optimizer=Adam(lr=lr_schedule(0)), loss=loss, metrics=[metrics.MeanSquaredError(),
                       metrics.MeanAbsoluteError(),
                       metrics.MeanAbsolutePercentageError(),
                       metrics.RootMeanSquaredError()])
time11=time()
print("jiazaishijain",time11-time00)
def pearson_r(y_true, y_pred):
    """
        皮尔逊相关系数，模型编译用
        """
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)
def bpcalculation(bprecording1,bprecording2,bprecording3,bprecording4,bprecording5):#,bprecording5
    """
       血压神经网络运算
        """
    bparr1 = np.array(bprecording1)
    bparr1 = np.expand_dims(bparr1, axis=0)
    bparr2 = np.array(bprecording2)
    bparr2 = np.expand_dims(bparr2, axis=0)
    bparr3 = np.array(bprecording3)
    bparr3 = np.expand_dims(bparr3, axis=0)
    bparr4 = np.array(bprecording4)
    bparr4 = np.expand_dims(bparr4, axis=0)
    bparr5 = np.array(bprecording5)
    bparr5 = np.expand_dims(bparr5, axis=0)
    print('bparr1.shape',bparr1.shape)
    bp = bpmodel.predict([bparr1,bparr2,bparr3,bparr4,bparr5])#,bparr5
    print('血压', bp[0])

    return bp[0]

time6=time()
BaseManager.register('get_task')
BaseManager.register('get_result')
conn = BaseManager(address = ('114.213.234.132',8786), authkey = b'123');
try:
    conn.connect();
except:
    print('连接失败');
    sys.exit();
task = conn.get_task();
result = conn.get_result();
while not task.empty():
    print('worker')
    n = task.get(timeout = 1);
    time7=time()
    print(n[0].shape)
    bp = bpcalculation(n[0],n[1],n[2],n[3],n[4])
    print('ok')
    time8=time()
    print("接受时间",time7-time6)
    print("分析时间",time8-time7)
    # print('run task %d' % n);
    sleeptime = random.randint(0,3);
    time.sleep(sleeptime);
    rt = (n, sleeptime);
    result.put(rt);
