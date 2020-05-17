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
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, metrics
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

plt.style.use('ggplot')

import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


missing_class = [0]
batch_size = 100
original_dim = 3072
latent_dim = 128 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256
epochs = 1000
num_classes = 10
FREEZE_LAYERS = 176

def remove_missclass(args):
    xdata, ydata, mc = args
    index = []
    for i in range(len(ydata)):
        if(ydata[i] in mc):
            index.append(i)
    xdata = np.delete(xdata, index, axis = 0)
    ydata = np.delete(ydata, index)

    return xdata, ydata

def reverse_remove_missclass(args):
    xdata, ydata, mc = args
    index = []
    for i in range(len(ydata)):
        if(ydata[i] not in mc):
            index.append(i)
    xdata = np.delete(xdata, index, axis = 0)
    ydata = np.delete(ydata, index)

    return xdata, ydata

def fill_miss_data_label(args):
    data, mc = args
    for i in range(len(mc)):
        data = np.insert(data, mc[i], values=0)
        
    return data

# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

#x_train, y_train_ = reverse_remove_missclass((x_train, y_train_, missing_class))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#y_train = to_categorical(y_train_, num_classes)
#y_test = to_categorical(y_test_, num_classes)
x_train_nonflat = x_train
x_test_nonflat = x_test
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


ye = load_model('cnn_miss_0/y_encoder.h5')
mean_miss = load_model('cnn_miss_0/mean_encoder.h5')
var_miss = load_model('cnn_miss_0/var_encoder.h5')

ym = ye.predict(np.eye(num_classes -1))

mean_miss = mean_miss.predict([x_test_nonflat], batch_size=batch_size)
var_miss = var_miss.predict([x_test_nonflat], batch_size=batch_size)

#-----------
#epsilon = np.random.normal(size=np.shape(mean_miss))
#data =  mean_miss + np.exp(var_miss / 2) * epsilon

def print_dis():
    for i in range(10):
        print(np.sum(np.abs(mean_miss[i] - mean_miss[i+1])))
        print(np.sum(np.abs(var_miss[i])))
        print(np.sum(np.exp(np.abs(var_miss[i]))))

def cal_acc_2():
    sum_class0_acc = 0
    sum_class1_9_acc = 0
    epsilon = np.random.normal(size=np.shape(mean_miss))
    
    for i in range(10):#range(len(y_test_)):
        if y_test_[i] == 0:  ##missing class
            bool_classification_error = False
            for j in range(len(ym)):
                print(np.sum(np.abs(ym[j] - mean_miss[i])), np.sum(np.abs(var_miss[i] / 2 *epsilon)))
                if np.sum(np.abs(ym[j] - mean_miss[i])) < np.sum(np.abs(var_miss[i] / 2 *epsilon)):   ##ym - mean = var * epsilon
                    bool_classification_error = True
            if bool_classification_error == False:
                sum_class0_acc += 1
            print('----------------------------------')
        else:  ##not missing class
            print(np.sum(np.abs(ym[y_test_[i] - 1] - mean_miss[i])), np.sum(np.abs(var_miss[i] / 2 *epsilon)))
            if np.sum(np.abs(ym[y_test_[i] - 1] - mean_miss[i])) < np.sum(np.abs(var_miss[i] / 2 *epsilon)):
                sum_class1_9_acc += 1
            print('=====================================')
                
    print(sum_class0_acc)
    print(sum_class1_9_acc)
    print((sum_class0_acc + sum_class1_9_acc) / len(y_test_))


def cal_acc():
    sum_class0_acc = 0
    sum_class1_9_acc = 0
    epsilon = np.random.normal(size=np.shape(mean_miss))
    vec = mean_miss + var_miss *epsilon

    miss_mean = []
    miss_var = []
    total = 0
    cnt = 0
    for i in range(len(y_test_)):
        if y_test_[i] in missing_class:  ##missing class
            miss_mean.append(mean_miss[i])
            miss_var.append(var_miss[i])
        else:  ##not missing class
            temp = []
            cnt += 1
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], vec[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(vec[i], vec[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])

                temp.append(cosine_similarity)
            temp = fill_miss_data_label([temp, missing_class])
            if np.argmax(temp) == y_test_[i]:
                total += temp[np.argmax(temp)]
                sum_class1_9_acc += 1

    cosine_thres = total / cnt  ##use known class cosine_similarity as upper bound
    epsilon = np.random.normal(size=np.shape(miss_mean[0]))
    for i in range(len(miss_mean)):
        bool_cosine_similarity_higher_than_thres = False
        vec = miss_mean[i] + miss_var[i] *epsilon
        for j in range(len(ym)):
            dot1 = np.dot(ym[j], vec)
            dot2 = np.dot(ym[j], ym[j])
            dot3 = np.dot(vec, vec)
            cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
            
            if cosine_similarity > cosine_thres:
                bool_cosine_similarity_higher_than_thres = True
                continue
        if bool_cosine_similarity_higher_than_thres == False:
            sum_class0_acc += 1
            
    print(sum_class0_acc)
    print(sum_class1_9_acc)
    print((sum_class0_acc + sum_class1_9_acc) / len(y_test_))

def cal_acc_extreme():
    sum_class0_acc = 0
    sum_class1_9_acc = 0
    epsilon = np.random.normal(size=np.shape(mean_miss))

    for i in range(len(y_test_)):
        if y_test_[i] == 0:  ##missing class
            bool_classification_error = False
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], mean_miss[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(mean_miss[i], mean_miss[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
                if cosine_similarity < 0.6:
                    bool_classification_error = True
                    continue
            if bool_classification_error == True:
                sum_class0_acc += 1

        else:  ##not missing class
            '''
            temp = []
            for j in range(len(ym)):
                cos = cosine(ym[j], mean_miss[i])
                temp.append(cos)
            if np.argmin(temp) == y_test_[i] - 1:
                sum_class1_9_acc += 1
            '''
            bool_classification_error = False
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], mean_miss[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(mean_miss[i], mean_miss[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
                if cosine_similarity < 0.6:
                    bool_classification_error = True
                    continue
            if bool_classification_error == True:
                sum_class1_9_acc += 1

    print(sum_class0_acc)
    print(sum_class1_9_acc)
    print((sum_class0_acc + sum_class1_9_acc) / len(y_test_))
    
def cal_acc_mean():
    sum_class0_acc = 0
    sum_class1_9_acc = 0
    epsilon = np.random.normal(size=np.shape(mean_miss))

    for i in range(len(y_test_)):
        temp = []
        if y_test_[i] == 0:  ##missing class
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], mean_miss[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(mean_miss[i], mean_miss[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
                temp.append(cosine_similarity)
            if np.mean(temp) < 0.5:
                sum_class0_acc += 1

        else:  ##not missing class
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], mean_miss[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(mean_miss[i], mean_miss[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
                temp.append(cosine_similarity)
            if np.mean(temp) > 0.5:
                sum_class1_9_acc += 1

    print(sum_class0_acc)
    print(sum_class1_9_acc)
    print((sum_class0_acc + sum_class1_9_acc) / len(y_test_))
    


def kmeans(data, y_test_):
    kmeans_fit = KMeans(n_clusters = 10).fit(data)
    cluster_labels = kmeans_fit.labels_
    y_test_ = y_test_.reshape(len(y_test_))
    cluster = [[], [], [], [], [], [], [], [], [], []]
    for i in range(len(cluster_labels)):
        cluster[cluster_labels[i]].append(y_test_[i])

    cluster_to_y_label = []
    cluster_to_y_label_intra_cluster_num = []
    for i in range(10):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(len(cluster[i])):
            temp[cluster[i][j]] = temp[cluster[i][j]] + 1
        cluster_to_y_label.append(np.argmax(temp))
        cluster_to_y_label_intra_cluster_num.append(temp[np.argmax(temp)])

    print(cluster_to_y_label)
    print(cluster_to_y_label_intra_cluster_num)
    temp = []
    for i in range(10):
        temp.append(len(cluster[i]))
    print(temp)


def test_cluster_num(data):
    silhouette_avgs = []
    ks = range(2, 15)
    for k in ks:
        kmeans_fit = cluster.KMeans(n_clusters = k).fit(data)
        cluster_labels = kmeans_fit.labels_
        silhouette_avg = metrics.silhouette_score(data, cluster_labels)
        silhouette_avgs.append(silhouette_avg)


    plt.bar(ks, silhouette_avgs)
    plt.show()
    print(silhouette_avgs)
    
#test_cluster_num(mean_miss)
kmeans(mean_miss, y_test_)
#print_dis()    
#cal_acc_mean()
#cal_acc_2()
#cal_acc()