# from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras.layers import *
from keras import backend as K
from keras.activations import sigmoid
import tensorflow as tf
# from keras.layers import Lambda
# from keras.layers import Layer
# from models.posatten import PAM
import numpy as np


def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block':  # CBAM_block
        net = cbam_block(net)
    elif attention_module == 'posattention':  # CBAM_block
        net = PAM()(net)
    # net = posattention(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


# class PAM_Module(Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = Parameter(torch.zeros(1))
#
#         self.softmax = Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out

""" Position attention module"""


# Ref from SAGAN
# def posattention(in_dim,input_feature):

def posattention(input_feature):
    """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
    """
    input_shape = input_feature.get_shape().as_list()
    _, h, w, filters = input_shape
    # print("filters",filters)

    b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)
    c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)
    d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input_feature)

    vec_b = K.reshape(b, (-1, h * w, filters // 8))
    vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
    bcT = K.batch_dot(vec_b, vec_cT)
    softmax_bcT = Activation('softmax')(bcT)
    vec_d = K.reshape(d, (-1, h * w, filters))
    bcTd = K.batch_dot(softmax_bcT, vec_d)
    bcTd = K.reshape(bcTd, (-1, h, w, filters))
    se = se_block(input_feature)
    out = bcTd + se
    print(out.shape)
    return out


def W_AveragePooling2D(x):
    # print("xxxxxx",x.shape)
    w_feature = Lambda(lambda x: K.mean(x, axis=1))
    w_feature = w_feature(x)
    print("wwwwwwww", w_feature.shape)
    return w_feature


def H_AveragePooling2D(x):
    # print("hhhhhhhh",x.shape)
    h_feature = Lambda(lambda x: K.mean(x, axis=2))
    h_feature = h_feature(x)
    print("hhhhhhhh", h_feature.shape)
    return h_feature


def se_block(input_feature, ratio=8):#8
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    # print("input_feature",input_feature.shape)
    print(K.image_data_format())
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    se_feature3 = GlobalAveragePooling2D()(input_feature)
    se_feature3 = Reshape((1, 1, channel))(se_feature3)
    assert se_feature3._keras_shape[1:] == (1, 1, channel)
    se_feature3 = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature3)
    assert se_feature3._keras_shape[1:] == (1, 1, channel // ratio)
    se_feature3 = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature3)
    se_feature0 = W_AveragePooling2D(input_feature)
    se_feature1 = H_AveragePooling2D(input_feature)
    # print(se_feature.dtype)
    se_feature0 = Reshape((1,input_feature.shape[2], channel))(se_feature0)
    se_feature1 = Reshape((input_feature.shape[1],1, channel))(se_feature1)
    # print(se_feature.dtype)
    assert se_feature0._keras_shape[1:] == (1,input_feature.shape[2], channel)
    se_feature0 = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature0)
    assert se_feature0._keras_shape[1:] == (1,input_feature.shape[2],channel//ratio)
    se_feature0 = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature0)
    assert se_feature1._keras_shape[1:] == (input_feature.shape[1],1, channel)
    se_feature1 = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature1)
    assert se_feature1._keras_shape[1:] == (input_feature.shape[1],1, channel // ratio)
    se_feature1 = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature1)
    assert se_feature0._keras_shape[1:] == (1,input_feature.shape[2], channel)
    if K.image_data_format() == 'channels_first':
        se_feature0 = Permute((3, 1,2))(se_feature0)#3,1,2
        print("after permute se_feature.shape",se_feature0.shape)
    se_feature0 = multiply([input_feature, se_feature0])
    se_feature1 = multiply([input_feature, se_feature1])
    se_feature3 = multiply([input_feature, se_feature3])
    se_feature = add([se_feature0,se_feature1,se_feature3])# ,
    # se_feature = spatial_attention((se_feature))

    # print("after multiply se_feature.shape", se_feature.shape)
    return se_feature

# def se_block(input_feature, ratio=8):  # 8
#     """Contains the implementation of Squeeze-and-Excitation(SE) block.
#     As described in https://arxiv.org/abs/1709.01507.
#     """
#     channel_axis = 1 if K.image_data_format() == "channels_first" else -1
#     channel = input_feature._keras_shape[channel_axis]
#     se_feature = GlobalAveragePooling2D()(input_feature)
#     se_feature = Reshape((1, 1, channel))(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel)
#     se_feature = Dense(channel // ratio,
#                        activation='relu',
#                        kernel_initializer='he_normal',
#                        use_bias=True,
#                        bias_initializer='zeros')(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
#     se_feature = Dense(channel,
#                        activation='sigmoid',
#                        kernel_initializer='he_normal',
#                        use_bias=True,
#                        bias_initializer='zeros')(se_feature)
#     assert se_feature._keras_shape[1:] == (1, 1, channel)
#     if K.image_data_format() == 'channels_first':
#         se_feature = Permute((3, 1, 2))(se_feature)
#
#     se_feature = multiply([input_feature, se_feature])
#     print("after multiply se_feature.shape", se_feature.shape)
#     return se_feature


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


# def channel_attention(input_feature, ratio=8):
#     channel_axis = 1 if K.image_data_format() == "channels_first" else -1
#     channel = input_feature._keras_shape[channel_axis]
#
#     shared_layer_one = Dense(channel // ratio,
#                              activation='relu',
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#     shared_layer_two = Dense(channel,
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#
#     avg_pool = W_AveragePooling2D(input_feature)
#     avg_pool1 = H_AveragePooling2D(input_feature)
#     avg_pool = Reshape((1, input_feature.shape[2], channel))(avg_pool)
#     avg_pool1 = Reshape((input_feature.shape[1],1, channel))(avg_pool1)
#     assert avg_pool._keras_shape[1:] == (1, input_feature.shape[2], channel)
#     avg_pool = shared_layer_one(avg_pool)
#     assert avg_pool1._keras_shape[1:] == (input_feature.shape[1],1, channel)
#     avg_pool1 = shared_layer_one(avg_pool1)
#     assert avg_pool._keras_shape[1:] == (1, input_feature.shape[2], channel // ratio)
#     avg_pool = shared_layer_two(avg_pool)
#     assert avg_pool1._keras_shape[1:] == (input_feature.shape[1],1, channel // ratio)
#     avg_pool1 = shared_layer_two(avg_pool1)
#     assert avg_pool._keras_shape[1:] == (1, input_feature.shape[2], channel)
#
#     max_pool = GlobalMaxPooling2D()(input_feature)
#     max_pool = Reshape((1, 1, channel))(max_pool)
#     assert max_pool._keras_shape[1:] == (1, 1, channel)
#     max_pool = shared_layer_one(max_pool)
#     assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
#     max_pool = shared_layer_two(max_pool)
#     assert max_pool._keras_shape[1:] == (1, 1, channel)
#
#     cbam_feature = Add()([avg_pool,avg_pool1, max_pool])
#     cbam_feature = Activation('sigmoid')(cbam_feature)
#
#     if K.image_data_format() == "channels_first":
#         cbam_feature = Permute((3, 1, 2))(cbam_feature)
#
#     return multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):

	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)

	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)

	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

