#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout
from keras.layers import Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


batch_size = 200
input_shape = (32, 32, 3)
kernel_size = 3
filters = 32
latent_dim = 128
intermediate_dim = 128
epochs = 1000
num_classes = 10
unseen_class = [0]
num_unseen_classes = 1

path = 'data/cifar10/'

seen_x = np.load(path+'seen_x.npy')
seen_y = np.load(path+'seen_y.npy')
unseen_x = np.load(path+'unseen_x.npy')
unseen_y = np.load(path+'unseen_y.npy')

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}    
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}

x_train, x_test, y_train, y_test = train_test_split(seen_x, seen_y, test_size=0.20, random_state=42)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

np.save(path+'x_train.npy', x_train)
np.save(path+'x_test.npy', x_test)
np.save(path+'y_train.npy', y_train)
np.save(path+'y_test.npy', y_test)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


temp = np.eye(num_classes).reshape(1,num_classes*num_classes)
y_class_train = np.repeat(a=temp, repeats=5000*(num_classes-num_unseen_classes), axis=0).reshape(5000*(num_classes-num_unseen_classes),num_classes,num_classes)
y_class_test =  np.repeat(a=temp, repeats=1000*(num_classes-num_unseen_classes), axis=0).reshape(1000*(num_classes-num_unseen_classes),num_classes,num_classes)


x_in = Input(shape=input_shape)
x = x_in
###############input_1#########################
mc1 = load_model('resnet101.h5')
mc1_x = mc1(x)
#mc1_x = Flatten()(mc1_x)
mc1_h = Dense(intermediate_dim, activation='relu')(mc1_x)
###############input_2#########################
for i in range(3):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               padding='same')(x)
    x = LeakyReLU(0.2)(x)

# 备份当前shape，等下构建decoder的时候要用
shape = K.int_shape(x)
x = Flatten()(x)
h = Dense(intermediate_dim, activation='relu')(x)

###################input_3#########################
y = Input(shape=(num_classes,)) # 输入类别
class_y = Input(shape=(num_classes,num_classes))
yh = Dense(latent_dim)(y) # 这里就是直接构建每个类别的均值
y_encoder = Model(y, yh)
y_mean = y_encoder(class_y)

def CosineLayer(x):    
    tot = 0
    for i in range(num_classes):
        for j in range(num_classes):

            dot1 = K.batch_dot(x[i], x[j], axes=1)
            dot2 = K.batch_dot(x[i], x[i], axes=1)
            dot3 = K.batch_dot(x[j], x[j], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            tot = tot + K.sum(dot1 / max_)
    return (tot - num_classes) / 2

cos_similarity  = Lambda(CosineLayer, output_shape=(1,))(y_mean) 

# 算p(Z|X)的均值和方差
mc1_z_mean = Dense(latent_dim)(mc1_h)
mc1_z_log_var = Dense(latent_dim)(mc1_h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

mc1_z_mean = BatchNormalization()(mc1_z_mean)
mc1_z_log_var = BatchNormalization()(mc1_z_log_var)
z_mean = BatchNormalization()(z_mean)
z_log_var = BatchNormalization()(z_log_var)

mc1_z_mean = Dropout(0.2)(mc1_z_mean)
mc1_z_log_var = Dropout(0.2)(mc1_z_log_var)
z_mean = Dropout(0.2)(z_mean)
z_log_var = Dropout(0.2)(z_log_var)

def parm_layer(args):
    m1, m2 = args
    return (m1 + m2) / 2

z_plus_mean = Lambda(parm_layer, output_shape=(latent_dim,))([z_mean, mc1_z_mean])
z_plus_log_var = Lambda(parm_layer, output_shape=(latent_dim,))([z_log_var, mc1_z_log_var])

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_plus_mean, z_plus_log_var])

# 解码层，也就是生成器部分
# 先搭建为一个独立的模型，然后再调用模型
latent_inputs = Input(shape=(latent_dim,))
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(3):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
    x = LeakyReLU(0.2)(x)
    filters //= 2

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same')(x)

# 搭建为一个独立的模型
decoder = Model(latent_inputs, outputs)

x_out = decoder(z)

# 建立模型
vae = Model(inputs=[x_in, y, class_y], outputs=[x_out, yh])

for layer in vae.layers:
    if layer.name == 'resnet101':
        layer.trainable = False


x_in_flat = Flatten()(x_in)
x_out_flat = Flatten()(x_out)
# xent_loss是重构loss，kl_loss是KL loss
xent_loss = 0.5 * K.sum(K.binary_crossentropy(x_in_flat, x_out_flat), axis=-1)#K.mean((x_in_flat - x_out_flat)**2)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_plus_log_var - K.square(z_plus_mean - yh) - K.exp(z_plus_log_var), axis=-1)

#cos相似度的loss,保證類別向量散度
cos_loss = -1 *cos_similarity#* 2e-5
vae_loss = K.mean(xent_loss + kl_loss + cos_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)



vae.fit([x_train, y_train, y_class_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test, y_class_test], None),
        #validation_split=(0.2),
        callbacks=[history, learning_rate_reduction, early_stopping])



history.loss_plot('epoch')

mean_encoder = Model(x_in, z_plus_mean)
mean_encoder.save('model/cifar10/mean_encoder.h5')

var_encoder = Model(x_in, z_plus_log_var)
var_encoder.save('model/cifar10/var_encoder.h5')

decoder.save('model/cifar10/generator.h5')

mu = Model(y, yh)
mu.save('model/cifar10/y_encoder.h5')
