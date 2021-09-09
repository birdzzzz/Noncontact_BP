from keras.models import *
from utils import lr_schedule
import os.path
import os
import numpy as np
import cv2
from keras import backend as K
from keras.layers import *
import pandas as pd
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras import metrics,losses
from keras.utils import multi_gpu_model
import tensorflow as tf
import csv
import glob
import dlib
import scipy.signal as signal
import scipy.fftpack as fftpack
from time import  *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"

def temporal_ideal_filter(sequence):
    print("sequence.shape",sequence.shape)
    results = []
    for i in range(3):
        fft = fftpack.fft(sequence[:,:,i], axis=0)
        frequencies = fftpack.fftfreq(sequence[:,:,i].shape[0], d=1.0 / 25)
        bound_low = (np.abs(frequencies - 0.75)).argmin()
        bound_high = (np.abs(frequencies - 2.5)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff = fftpack.ifft(fft, axis=0)
        result = np.abs(iff)
        results.append(result)
    results = np.array(results)
    results = np.moveaxis(results,0,2)
    print("results",results.shape)
    return results
def pearson_r(y_true, y_pred):

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



######增加维度######
def bpcalculation(bprecording1,bprecording2,bprecording3,bprecording4,bprecording5):#,bprecording5
    """
       血压神经网络运�?
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
    bp = bpmodel.predict([bparr1,bparr2,bparr3,bparr4,bparr5])#,bparr5
    print('血', bp[0])

    return bp[0]
def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    newsignals = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        # for j in range(3):
        sig = signal.lfilter(b, a, data[:,i, :])
        newsignals[:, i, :] = sig
    # return signal.lfilter(b, a, data)
    return  newsignals
def loadlabel(labelpath):
    recording=[]
    with open(labelpath, 'r',  encoding='UTF-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            recording.append(float(row[0]))
            recording.append(float(row[1]))
    recording=np.array(recording)
    return recording



def loadbmi(bmipath,num):
    recording = []
    # print(num)
    with open(bmipath, 'r', encoding='UTF-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        #print(reader)
        for row in reader:
            # print(row[0])
            if int(row[0])==int(num):
                # print("row[0]",row[0])
                recording.append(float(row[3]))
                recording.append(float(row[4]))
                recording.append(float(row[2]))
        # print(row)
    # print("recording",recording)
    height=recording[0]
    weight=recording[1]
    # print(weight)
    bmi=weight/np.power(height,2)
    recording.append(bmi)
    recording= np.array(recording)
    # print(recording)
    return recording

def bland_altman_plot(data1, data2, *args, **kwargs):
    # data1 =np.asarray(data1)
    # data2 = np.asarray(data2)
    mean= np.mean([data1, data2], axis=0)
    diff= data1 - data2                   # Difference between data1 and data2
    md= np.mean(diff)                   # Mean of the difference
    sd= np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, c='r', s=50, alpha=0.5,*args, **kwargs)
    plt.axhline(md,color='black', linestyle='--',linewidth=2)
    plt.axhline(md + 1.96*sd, color='black', linestyle='--', linewidth=2)
    plt.axhline(md - 1.96*sd, color='black', linestyle='--', linewidth=2)

    plt.xlabel('$(SBP_{gt}+SBP_{predict})/2$', fontproperties='Times New Roman', fontsize=10)
    plt.ylabel('$SBP_{gt}-SBP_{predict}$', fontproperties='Times New Roman', fontsize=10)
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)

    plt.text(130, 0, "mean", fontproperties='Times New Roman', size=12,color='black')
    plt.text(130, 12, "mean+1.96*sd", fontproperties='Times New Roman', size=12, color='black')
    plt.text(130, -12, "mean-1.96*sd", fontproperties='Times New Roman', size=12, color='black')
def mapee(y_true, y_pred):

    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape
def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)


def loaddata2(path):
    labelarr1 = []
    bparr =[]
    bmipath = r'/home/som/lab/jmr/BPCEM/info1.csv'
    date = os.listdir(path)
    print("bmipath")

    for i in range(len(date)):

        datafolder = os.path.join(path, date[i])
        # print(datafolder)
        vonlunteers = os.listdir(datafolder)
        for j in range(len(vonlunteers)):
            sequence=[]
            sequence1 = []
            sequence2 = []
            sequence3=[]

            labelpath = datafolder + '/' + vonlunteers[j] + '/' + 'bp.csv'
            imgpath = datafolder + '/' + vonlunteers[j]

            for k in range(200,500):

                nose_image = os.path.join(imgpath, 'nose_{0}.jpg'.format(k))
                forehead_image = os.path.join(imgpath, 'forehead_{0}.jpg'.format(k))
                hand_image = os.path.join(imgpath, 'hand_{0}.jpg'.format(k-200))
                nose_roi = cv2.imread(nose_image)
                fh_roi = cv2.imread(forehead_image)
                try:
                    hand_roi = cv2.imread(hand_image)
                    if hand_roi.shape[0]>80 and hand_roi.shape[1]>80:
                        hand_roi=hand_roi[hand_roi.shape[0]-80:hand_roi.shape[0],hand_roi.shape[1]-80:hand_roi.shape[1]]

                except:
                    print(hand_image,"hand_roi无法读取")
                    break
                # print(nose_image,"start")
                roi3 = cv2.resize(hand_roi, (80, 80))
                roi2 = cv2.resize(fh_roi, (80, 50))
                roi1 = cv2.resize(nose_roi, (80, 30))
                roi = np.vstack((roi1, roi2))
                # print("roi",roi.shape)
                try:

                    gussimage = roi.copy()
                    pyramid = [gussimage]
                    for n in range(3):
                        gussimage = cv2.pyrDown(gussimage)
                        pyramid.append(gussimage)
                    gaussian_frame1 = pyramid[-1]
                    gaussian_frame2 = pyramid[-2]
                    gaussian_frame1 = np.array(gaussian_frame1)
                    gaussian_frame2 = np.array(gaussian_frame2)
                    gaussian_frame1 = gaussian_frame1.reshape(100,3)
                    gaussian_frame2 = gaussian_frame2.reshape(400,3)

                except:
                    print('￥￥￥￥￥roi无法检测￥￥￥￥￥')
                try:

                    gussimage_hand = roi3.copy()
                    pyramid_hand = [gussimage_hand]
                    for n in range(3):
                        gussimage_hand = cv2.pyrDown(gussimage_hand)
                        pyramid_hand.append(gussimage_hand)
                    gaussian_frame1_hand = pyramid_hand[-1]
                    gaussian_frame2_hand = pyramid_hand[-2]
                    gaussian_frame1_hand = np.array(gaussian_frame1_hand)
                    gaussian_frame2_hand = np.array(gaussian_frame2_hand)
                    gaussian_frame1_hand = gaussian_frame1_hand.reshape(100, 3)
                    gaussian_frame2_hand = gaussian_frame2_hand.reshape(400, 3)



                except:
                    print('￥￥￥￥￥roi无法检测￥￥￥￥￥')
                    break
                sequence.append(gaussian_frame1)
                sequence1.append(gaussian_frame2)
                sequence2.append(gaussian_frame1_hand)
                sequence3.append(gaussian_frame2_hand)  # 300,80,80,3



            bmirecording = loadbmi(bmipath, vonlunteers[j].split('_')[0])
            labelrecording = loadlabel(labelpath)
            labelarr1.append(labelrecording)

            sequence = butterworth_filter(np.array(sequence), 0.75, 2.5, 25)
            sequence1 = butterworth_filter(np.array(sequence1), 0.75, 2.5, 25)
            sequence2 = butterworth_filter(np.array(sequence2), 0.75, 2.5, 25)
            sequence3 = butterworth_filter(np.array(sequence3), 0.75, 2.5, 25)

            bp = bpcalculation(sequence,sequence1,sequence2,sequence3,bmirecording)
            bparr.append(bp)
    # bland_altman_plot(np.array(labelarr1)[:, 0], np.array(bparr)[:,0])
    # plt.show()
    labelarr = np.array(labelarr1)[:,1]
    predict_arr = np.array(bparr)[:,1]
    print("shape",predict_arr.shape)


    mse = mean_squared_error(labelarr, predict_arr)
    mae = mean_absolute_error(labelarr, predict_arr)
    mape = mapee(np.array(labelarr), np.array(predict_arr))
    r = pearsonr(labelarr, predict_arr)
    print('MSE:', mse)
    print('MAE:', mae)
    print('MAPE:', mape)
    print('RMSE:', np.sqrt(mse))
    print('R:', r)
    print(stats.spearmanr(predict_arr, labelarr)[0])
    print(stats.stats.kendalltau(predict_arr ,labelarr)[0])

    return 0

time1=time()
loss = losses.mean_squared_error
modelpath='/home/som/lab/jmr/BPCEM/saved_models/v2BP715.hdf5'##########################################################################################################
bpmodel=load_model(modelpath,compile=False)
bpmodel.summary()
bpmodel.compile(optimizer=Adam(lr=lr_schedule(0)), loss=loss, metrics=[metrics.MeanSquaredError(),
                       metrics.MeanAbsoluteError(),
                       metrics.MeanAbsolutePercentageError(),
                       metrics.RootMeanSquaredError(), pearson_r])
time3=time()
loaddata2("/home/som/lab-data/jmrnewdata/laboratorydata/test/")#
time2=time()
print("运行时间",(time2-time1)/160)
print("加载时间",(time3-time1))


