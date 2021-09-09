import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras import metrics,losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from models import resnext, resnet_v1, resnet_v2, mobilenets, inception_v3, inception_resnet_v2, densenet
import numpy as np
import os
import os.path
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
import tensorflow as tf
import csv
import glob
import cv2
import time

import scipy.signal as signal
import scipy.fftpack as fftpack
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


batch_size = 8
epochs =130
data_augmentation = False
subtract_pixel_mean = True
base_model = 'resnet20'
attention_module = 'se_block'
model_type = base_model if attention_module==None else base_model+'_'+attention_module
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
def loadbmi(bmipath,num):
    recording = []

    with open(bmipath, 'r', encoding='UTF-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if int(row[0])==int(num):
                recording.append(float(row[3]))
                recording.append(float(row[4]))
                recording.append(float(row[2]))
                break
    print("recording%%%%%%%%",num,bmipath,recording)

    height=recording[0]
    weight=recording[1]

    bmi=weight/np.power(height,2)
    recording.append(bmi)
    recording= np.array(recording)
    return recording

def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
def loadlabel(labelpath):
    recording=[]
    with open(labelpath, 'r',  encoding='UTF-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if float(row[0])<float(row[1]):
                t = row[0]
                row[0] = row[1]
                row[1] = t
            recording.append(float(row[0]))
            recording.append(float(row[1]))

    return recording

def butterworth_filter(data, low, high, sample_rate, order=5):

    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')

    newsignals=np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        # for j in range(3):
        sig = signal.lfilter(b, a, data[:,i,:])
        newsignals[:,i,:]=sig
    return newsignals
def load_data(path): ###归一化
    bmiarr=[]
    imgsarr = []
    imgsarr1 = []
    imgsarr2=[]
    imgsarr3 = []
    labelarr = []


    bmipath = r'/home/som/lab/jmr/BPCEM/info1.csv'
    date = os.listdir(path)

    for i in range(len(date)):
        datafolder = os.path.join(path, date[i])
        vonlunteers = os.listdir(datafolder)
        for j in range(len(vonlunteers)):
            imgpath = datafolder + '/' + vonlunteers[j]
            labelpath = datafolder + '/' + vonlunteers[j] + '/' + 'bp.csv'
            print(labelpath)
            images=[]
            sequence=[]
            sequence1 = []
            sequence2 = []
            sequence3 = []
            num = 0
            testpath = os.path.join(imgpath, 'hand_1.jpg')
            if not os.path.exists(testpath):
                break
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
                    break

                roi3 = cv2.resize(hand_roi,(80, 80))
                roi2 = cv2.resize(fh_roi, (80, 50))
                roi1 = cv2.resize(nose_roi, (80, 30))
                roi = np.vstack((roi1, roi2))
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
                    print('￥￥￥￥￥roi无法检测￥￥￥￥￥￥')
                    break
                try:

                        gussimage_hand = roi3.copy()
                        pyramid_hand = [gussimage_hand]
                        for n in range(3):
                            gussimage_hand = cv2.pyrDown(gussimage_hand)
                            pyramid_hand.append(gussimage_hand)
                        gaussian_frame1_hand = pyramid_hand[-1]
                        gaussian_frame2_hand = pyramid_hand[-2]
                        gaussian_frame1_hand = np.array(gaussian_frame1_hand)
                        # print("gaussian_frame1",gaussian_frame1.shape)#10 10 3
                        gaussian_frame2_hand = np.array(gaussian_frame2_hand)
                        # print("gaussian_frame2", gaussian_frame2.shape)#20 20 3
                        gaussian_frame1_hand = gaussian_frame1_hand.reshape(100,3)
                        gaussian_frame2_hand = gaussian_frame2_hand.reshape(400,3)



                except:
                    print('￥￥￥￥￥roi无法检测￥￥￥￥￥￥')
                    break
                sequence.append(gaussian_frame1)
                sequence1.append(gaussian_frame2)
                sequence2.append(gaussian_frame1_hand)#300,80,80,3#huifu
                sequence3.append(gaussian_frame2_hand)  # 300,80,80,3#huifu
                images.append(roi)
                num = num +1

            print("vonlunteers[j].split('_')[0]",vonlunteers[j].split('_')[0])

            bmirecording = loadbmi(bmipath, vonlunteers[j].split('_')[0])
            bmiarr.append(bmirecording)
            labelrecording = loadlabel(labelpath)
            labelarr.append(labelrecording)


            sequence = butterworth_filter(np.array(sequence),0.75,2.5,25)
            sequence1 = butterworth_filter(np.array(sequence1),0.75,2.5,25)
            sequence2 = butterworth_filter(np.array(sequence2),0.75,2.5,25)
            sequence3 = butterworth_filter(np.array(sequence3),0.75,2.5,25)


            imgsarr.append(sequence)
            imgsarr1.append(sequence1)
            imgsarr2.append(sequence2)
            imgsarr3.append(sequence3)

    imgsarr=np.array(imgsarr)
    imgsarr1 = np.array(imgsarr1)
    imgsarr2 = np.array(imgsarr2)
    imgsarr3 = np.array(imgsarr3)



    bmiarr = np.array(bmiarr)
    print("bmiarr",bmiarr.shape)
    labelarr=np.array(labelarr)

    return imgsarr, imgsarr1, imgsarr2,imgsarr3, labelarr, bmiarr




datapath = r'/home/som/lab-data/jmrnewdata/laboratorydata/'

train_path=os.path.join(datapath,'train')
val_path=os.path.join(datapath,'val')
test_path=os.path.join(datapath,'test')

x_train,x1_train,x2_train,x3_train,y_train, bmi_train=load_data(train_path)
print('训练集数据维度如下!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(x_train.shape)
print(y_train.shape)

x_val,x1_val,x2_val,x3_val,y_val, bmi_val=load_data(val_path)

print('验证集数据维度如下!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(x_val.shape)
print(y_val.shape)
x_test,x1_test,x2_test,x3_test,y_test, bmi_test=load_data(test_path)

print('测试集数据维度如下!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(x_test.shape)
print(y_test.shape)
input_shape = x_train.shape[1:]
input2_shape = x1_train.shape[1:]
input_shape_hand = x2_train.shape[1:]
input2_shape_hand = x3_train.shape[1:]


print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",type(input_shape[1]))
print("input_shape",input_shape)

start = time.time()
depth = 20
model = resnet_v1.resnet_v1_hand(input_shape=input_shape,input2_shape=input2_shape,input_shape_hand=input_shape_hand,input2_shape_hand=input2_shape_hand,depth=depth,bmi_train=bmi_train,attention_module=attention_module)

loss = losses.mean_squared_error
model = multi_gpu_model(model, gpus=2)
model.summary()
model.compile(optimizer=Adam(lr=lr_schedule(0)), loss=loss, metrics=[metrics.MeanSquaredError(),
                       metrics.MeanAbsoluteError(),
                       metrics.MeanAbsolutePercentageError(),
                       metrics.RootMeanSquaredError(), pearson_r])

print(model_type)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'v2BP715.hdf5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False)
print('!'*50,filepath)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
csv_logger = CSVLogger('v2addd_715.csv')
callbacks = [checkpoint, lr_reducer, lr_scheduler,csv_logger]

model.fit([x_train,x1_train,x2_train,x3_train,bmi_train], y_train, batch_size=batch_size, epochs=epochs,validation_data=([x_val,x1_val,x2_val,x3_val,bmi_val], y_val),shuffle=True,callbacks=callbacks)
end = time.time()
scores = model.evaluate([x_val,x1_val,x2_val,x3_val,bmi_val], y_val, verbose=1)
print("time",end-start)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])