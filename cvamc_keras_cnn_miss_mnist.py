#! -*- coding: utf-8 -*-


'''用Keras实现的CVAE
   目前只保证支持Tensorflow后端

 #来自
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout
from keras.layers import Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import mnist
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

def remove_missclass(args):
    xdata, ydata, mc = args
    index = []
    for i in range(len(ydata)):
        if(ydata[i] in mc):
            index.append(i)
    xdata = np.delete(xdata, index, axis = 0)
    ydata = np.delete(ydata, index)

    return xdata, ydata

batch_size = 200
input_shape = (28, 28, 1)
kernel_size = 3
filters = 64
latent_dim = 128
intermediate_dim = 256
epochs = 1000
num_classes = 10

missing_class = [0]



(x_train, y_train_), (x_test, y_test_) = mnist.load_data()


#remove class #
x_train, y_train_ = remove_missclass((x_train, y_train_, missing_class))
x_test, y_test_ = remove_missclass((x_test, y_test_, missing_class))
num_classes = num_classes -1
y_train_ = y_train_ - 1
y_test_ = y_test_ -1

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)


temp = np.eye(num_classes).reshape(1,num_classes*num_classes)
y_class_train = np.repeat(a=temp, repeats=54077, axis=0).reshape(54077,num_classes,num_classes)
y_class_test = np.repeat(a=temp, repeats=9020, axis=0).reshape(9020,num_classes,num_classes)


'''

'''
'''

mc1 = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
mc1_x = mc1.output
mc1_x = Flatten()(mc1_x)
mc1_h = Dense(intermediate_dim, activation='relu')(mc1_x)
'''


x_in = Input(shape=input_shape)
x = x_in
###############input_1#########################
mc1 = load_model('model-mnist-pretrain.h5')
intermediate_mc1 = Model(inputs=mc1.input, outputs=mc1.get_layer('cnn_output').output, name='intermediate_mc1')
mc1_x = intermediate_mc1(x)
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
    for i in range(9):
        for j in range(9):
            '''
            dot1 = K.batch_dot(K.reshape(x[i], (10,128)), K.reshape(x[j], (128,10)), axes=1)
            dot2 = K.batch_dot(K.reshape(x[i], (10,128)), K.reshape(x[i], (128,10)), axes=1)
            dot3 = K.batch_dot(K.reshape(x[j], (10,128)), K.reshape(x[j], (128,10)), axes=1)
            '''
            dot1 = K.batch_dot(x[i], x[j], axes=1)
            dot2 = K.batch_dot(x[i], x[i], axes=1)
            dot3 = K.batch_dot(x[j], x[j], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            tot = tot + K.sum(dot1 / max_)
    return (tot - 9) / 2

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

# 重参数技巧
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
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(3):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
    x = LeakyReLU(0.2)(x)
    filters //= 2

x = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='relu',
                          padding='same')(x)

x = Flatten()(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
outputs = Dense(28 * 28 * 1, activation='sigmoid')(x)
# 搭建为一个独立的模型
decoder = Model(latent_inputs, outputs)


x_out = decoder(z)

# 建立模型
vae = Model(inputs=[x_in, y, class_y], outputs=[x_out, yh])

for layer in vae.layers:
    if layer.name == 'intermediate_mc1':
        layer.trainable = False


x_in_flat = Flatten()(x_in)
x_out_flat = x_out
# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x_in_flat, x_out_flat), axis=-1)#0.5 * K.mean((x_in_flat - x_out_flat)**2)#
# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_plus_log_var - K.square(z_plus_mean - yh) - K.exp(z_plus_log_var), axis=-1)

#cos相似度的loss,保證類別向量散度
cos_loss = cos_similarity#* 2e-5
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
mean_encoder.save('cnn_miss_0/mean_encoder_hidedim_64.h5')

var_encoder = Model(x_in, z_plus_log_var)
var_encoder.save('cnn_miss_0/var_encoder_hidedim_64.h5')

decoder.save('cnn_miss_0/generator_hidedim_64.h5')

mu = Model(y, yh)
mu.save('cnn_miss_0/y_encoder_hidedim_64.h5')

'''
# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 输出每个类的均值向量
mu = Model(y, yh)
mu = mu.predict(np.eye(num_classes))

# 观察能否通过控制隐变量的均值来输出特定类别的数字
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

output_digit = 9 # 指定输出数字

# 用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][1]
grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][0]

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
'''