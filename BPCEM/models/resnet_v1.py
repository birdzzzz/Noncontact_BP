"""
ResNet v1
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers import *
# from FFT import HR_BP_RR
# from keras.layers import *
from models.attention_module import attach_attention_module
# def loadbmi(bmipath,num):
#     recording = []
#     # print(num)
#     with open(bmipath, 'r', encoding='UTF-8-sig') as csvfile:
#         reader = csv.reader(csvfile)
#         #print(reader)
#         for row in reader:
#             if row[0]==num:
#                 recording.append(float(row[1]))
#                 recording.append(float(row[2]))
#         # print(row)
#         # print(recording)
#     height=recording[0]
#     weight=recording[1]
#     # print(weight)
#     bmi=weight/np.power(height,2)
#     recording.append(bmi)
#     recording= np.array(recording)
#     # print(recording)
#     return recording
# def bmi_data(path): ###归一化
#     bmiarr=[]
#     bmipath = r'/home/som/lab/jmr/five-video-classification-methods-master/1.csv'
#     date = os.listdir(path)
#     for i in range(len(date)):
#         datafolder = os.path.join(path, date[i])
#         vonlunteers = os.listdir(datafolder)
#         for j in range(len(vonlunteers)):
#             bmirecording = loadbmi(bmipath, vonlunteers[j].split('-')[0])
#             bmiarr.append(bmirecording)
#     bmiarr = np.array(bmiarr)
#     return bmiarr####
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,#1
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# def resnet_v1(input_shape, depth, attention_module=None):
def resnet_v1(input_shape, depth,  attention_module=None):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    print("^^^^^^^^^^^^^^^^^^^^",type(input_shape[1]))
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    inputs = Input(shape=input_shape)
    # bmi_inputs = Input(shape=(bmi_train.shape[1],))#########################################


    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1#1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 3  # downsample  2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack??????????????????????????????
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,#1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            # attention_module
            if attention_module is not None:
                y = attach_attention_module(y, attention_module)
                print("attach_attention_module(y, attention_module)",y.shape)
            x = keras.layers.add([x, y])#直接对张量求和????????????????????????????????????
            x = Activation('relu')(x)

        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=4)(x)#8
    y = Flatten()(x)
    print("flatten",y.shape)

    # y = Dense(32, activation='relu', )(y)
    # y = concatenate([y, fft_inputs])
    # y = concatenate([y, bmi_inputs])
    # y = concatenate([y, y2,y3])
    # y = Dense(32, activation='relu', )(y)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)
    # y = concatenate([y, inputbmi])
    # y = Dense(32, activation='relu')(y)
    outputs = Dense(2, activation='relu')(y)

    # Instantiate model.
    # model = Model(inputs=[inputs,bmi_inputs], outputs=outputs)
    model = Model(inputs=inputs, outputs=outputs)
    # model.summary()
    # model = Model(inputs=inputs, outputs=outputs)
    return model
def resnet_v1_multi(input_shape1, input_shape2,depth,attention_module=None):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    print("^^^^^^^^^^^^^^^^^^^^",type(input_shape1[1]))
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    inputs = Input(shape=input_shape1)
    inputs2 = Input(shape=input_shape2)
    # bmi_inputs = Input(shape=(bmi_train.shape[1],))

    # inputbmi = Input(shape=(bmi_train.shape[1],))
    x = resnet_layer(inputs=inputs)
    x2 = resnet_layer(inputs=inputs2)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1#1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 3  # downsample  2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y2 = resnet_layer(inputs=x2,
                             num_filters=num_filters,
                             strides=strides)


            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            y2 = resnet_layer(inputs=y2,
                             num_filters=num_filters,
                             activation=None)


            if stack > 0 and res_block == 0:  # first layer but not first stack??????????????????????????????
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,#1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                x2 = resnet_layer(inputs=x2,
                                 num_filters=num_filters,
                                 kernel_size=1,  # 1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)


            # attention_module
            if attention_module is not None:
                y = attach_attention_module(y, attention_module)
                y2 = attach_attention_module(y2, attention_module)

                print("attach_attention_module(y, attention_module)",y.shape)
            x = keras.layers.add([x, y])#直接对张量求和????????????????????????????????????
            x2 = keras.layers.add([x2, y2])

            x = Activation('relu')(x)
            x2 = Activation('relu')(x2)


        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=4)(x)#8
    x2 = AveragePooling2D(pool_size=4)(x2) #8

    y = Flatten()(x)
    y2 = Flatten()(x2)

    print("flatten",y.shape)

    # y = Dense(32, activation='relu', )(y)
    # y = concatenate([y, fft_inputs])
    y = concatenate([y,y2])
    # y = concatenate([y, y2,y3])
    # y = Dense(32, activation='relu', )(y)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)
    # y = concatenate([y, inputbmi])
    # y = Dense(32, activation='relu')(y)
    outputs = Dense(2, activation='relu')(y)

    # Instantiate model.
    model = Model(inputs=[inputs,inputs2], outputs=outputs)
    # model = Model(inputs=inputs, outputs=outputs)
    # model.summary()
    # model = Model(inputs=inputs, outputs=outputs)
    return model
def resnet_v1_hand(input_shape, input2_shape,input_shape_hand,input2_shape_hand,depth, bmi_train, attention_module=None):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    print("^^^^^^^^^^^^^^^^^^^^",type(input_shape[1]))
    # print("*******************bmi.shape",bmi_train.shape)
    # print("*******************r.shape", r_train.shape)
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    inputs = Input(shape=input_shape)
    inputs2 = Input(shape=input2_shape)
    bmi_inputs = Input(shape=(bmi_train.shape[1],))###############################################################################huifu
    hand_inputs = Input(shape=input_shape_hand)
    hand_inputs2 = Input(shape=input2_shape_hand)
    # face_hand_inputs = Input(shape=input_shape)
    # inputbmi = Input(shape=(bmi_train.shape[1],))
    x = resnet_layer(inputs=inputs)
    x1 = resnet_layer(inputs=inputs2)
    x2=resnet_layer(inputs=hand_inputs)
    x3 = resnet_layer(inputs=hand_inputs2)
    # x4 = resnet_layer(inputs=fh_mean)
    # x5= resnet_layer(inputs=hand_mean)
    # x4_x5 = keras.layers.subtract([x4, x5])
    # x4_x5 = keras.layers.subtract([fh_mean, hand_mean])
    x_x2=keras.layers.subtract([x, x2])
    x1_x3 = keras.layers.subtract([x1, x3])
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1#1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 3  # downsample  2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y1 = resnet_layer(inputs=x1,
                             num_filters=num_filters,
                             strides=strides)
            y2 = resnet_layer(inputs=x2,
                             num_filters=num_filters,
                             strides=strides)
            y3 = resnet_layer(inputs=x3,
                             num_filters=num_filters,
                             strides=strides)
            y_y2 = resnet_layer(inputs=x_x2,
                             num_filters=num_filters,
                             strides=strides)
            y1_y3 = resnet_layer(inputs=x1_x3,
                                num_filters=num_filters,
                                strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            y1 = resnet_layer(inputs=y1,
                             num_filters=num_filters,
                             activation=None)
            y2 = resnet_layer(inputs=y2,
                             num_filters=num_filters,
                             activation=None)
            y3 = resnet_layer(inputs=y3,
                              num_filters=num_filters,
                              activation=None)
            y_y2 = resnet_layer(inputs=y_y2,
                             num_filters=num_filters,
                             activation=None)
            y1_y3 = resnet_layer(inputs=y1_y3,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack????
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,#1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                x1 = resnet_layer(inputs=x1,
                                 num_filters=num_filters,
                                 kernel_size=1,  # 1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                x2 = resnet_layer(inputs=x2,
                                 num_filters=num_filters,
                                 kernel_size=1,  # 1
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                x3 = resnet_layer(inputs=x3,
                                  num_filters=num_filters,
                                  kernel_size=1,  # 1
                                  strides=strides,
                                  activation=None,
                                  batch_normalization=False)
                x_x2= resnet_layer(inputs=x_x2,
                             num_filters=num_filters,
                             kernel_size=1,  # 1
                             strides=strides,
                             activation=None,
                             batch_normalization=False)
                x1_x3 = resnet_layer(inputs=x1_x3,
                                    num_filters=num_filters,
                                    kernel_size=1,  # 1
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
            # attention_module
            if attention_module is not None:
                y = attach_attention_module(y, attention_module)
                y1 = attach_attention_module(y1, attention_module)
                print("attach_attention_module(y, attention_module)",y.shape)
                y2 = attach_attention_module(y2, attention_module)
                print("attach_attention_module(y2, attention_module)", y2.shape)
                y3 = attach_attention_module(y3, attention_module)
                y_y2 = attach_attention_module(y_y2, attention_module)
                y1_y3 = attach_attention_module(y1_y3, attention_module)
            x = keras.layers.add([x, y])#直接对张量求和
            x1 = keras.layers.add([x1, y1])
            x2 = keras.layers.add([x2, y2])  # 直接对张量求和
            x3 = keras.layers.add([x3, y3])
            x_x2=keras.layers.add([x_x2, y_y2])
            x1_x3 = keras.layers.add([x1_x3, y1_y3])
            # x_x2 = keras.layers.subtract([x, x2])
            x = Activation('relu')(x)
            x1 = Activation('relu')(x1)
            x2 = Activation('relu')(x2)
            x3 = Activation('relu')(x3)
            x_x2 = Activation('relu')(x_x2)
            x1_x3 = Activation('relu')(x1_x3)
        num_filters *= 2
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=4)(x)#8
    x1 = AveragePooling2D(pool_size=4)(x1)  # 8
    x2 = AveragePooling2D(pool_size=4)(x2) # 8
    x3 = AveragePooling2D(pool_size=4)(x3)  # 8
    x_x2 = AveragePooling2D(pool_size=4)(x_x2)
    x1_x3 = AveragePooling2D(pool_size=4)(x1_x3)  # 8
    y = Flatten()(x)
    y1 = Flatten()(x1)
    y2 = Flatten()(x2)
    y3 =Flatten()(x3)
    y_y2 = Flatten()(x_x2)
    y1_y3 = Flatten()(x1_x3)
    print("flatten",y.shape)
    print("flatten2", y2.shape)
    print("flatten3", y3.shape)
    # y = Dense(32, activation='relu', )(y)
    # y = concatenate([y, fft_inputs])
    # y = concatenate([y, inputbmi])
    # y = concatenate([y,y1, y2,y3,y_y2,y1_y3])
    #全部保留7.48
    output = Dense(32,activation='relu')(y)#face
    output1 = Dense(32, activation='relu')(y1)#face去掉后
    output2 = Dense(32, activation='relu')(y2)#hand去掉后loss不下降
    output3 = Dense(32, activation='relu')(y3)#hand去掉后7.52
    output4 = Dense(32, activation='relu')(y_y2)#face-hand去掉后8.04  不下降
    output5 = Dense(32, activation='relu')(y1_y3)#face-hand去掉后loss三千多不下降
    outputs = concatenate([output,output1,output2,output3,output4,output5,bmi_inputs])#,bmi_inputs###############################################huifu
    outputs = Dense(2, activation='relu')(outputs)
    # Instantiate model.
    model = Model(inputs=[inputs,inputs2,hand_inputs,hand_inputs2,bmi_inputs], outputs=outputs)#,bmi_inputs#######################################huifu

    # model.summary()

    return model