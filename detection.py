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

unseen_class = [0]


def print_dis(mean, var):
    for i in range(10):
        print(np.sum(np.abs(mean[i] - mean[i+1])))
        print(np.sum(np.abs(var[i])))
        #print(np.sum(np.exp(np.abs(var[i]))))

def y_posterior(label, ym, mean, var):
    sum_unseen = 0
    sum_seen = 0
    epsilon = np.random.normal(size=np.shape(mean))
    bool_unseen = False
    if label[0] in unseen_class:
        bool_unseen = True
    
    if bool_unseen == True:
        for i in range(len(label)):
            bool_classification_error = False
            for j in range(len(ym)):
                if np.sum(np.abs(ym[j] - mean[i])) < np.sum(np.abs(var[i] / 2 *epsilon)):   ##ym - mean = var * epsilon
                    bool_classification_error = True
            if bool_classification_error == False:
                sum_unseen += 1
        return sum_unseen
    else:
        for i in range(len(label)):
            if np.sum(np.abs(ym[label[i]] - mean[i])) < np.sum(np.abs(var[i] / 2 *epsilon)):
                sum_seen += 1
                
        return sum_seen


def y_cosine_similarity(label, ym, mean, var, cosine_thres=-1):
    sum_unseen = 0
    sum_seen = 0
    epsilon = np.random.normal(size=np.shape(mean))
    vec = mean + var *epsilon

    if cosine_thres == -1:
        total = 0
        for i in range(len(label)):
            temp = []
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], vec[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(vec[i], vec[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])

                temp.append(cosine_similarity)
            if np.argmax(temp) == label[i]:
                total += temp[np.argmax(temp)]
                sum_seen += 1
        cosine_thres = total / len(label)
        return sum_seen, cosine_thres
    else:
        for i in range(len(label)):
            bool_cosine_similarity_higher_than_thres = False
            for j in range(len(ym)):
                dot1 = np.dot(ym[j], vec[i])
                dot2 = np.dot(ym[j], ym[j])
                dot3 = np.dot(vec[i], vec[i])
                cosine_similarity = dot1 / np.max([np.sqrt(dot2 * dot3), 1e-5])
            
                if cosine_similarity > cosine_thres:
                    bool_cosine_similarity_higher_than_thres = True
                    break
            if bool_cosine_similarity_higher_than_thres == False:
                sum_unseen += 1
            
        return sum_unseen


def kmeans(data, label, cluster_num):
    kmeans_fit = KMeans(n_clusters = cluster_num).fit(data)
    cluster_labels = kmeans_fit.labels_
    label = label.reshape(len(label))
    cluster = [[] for x in range(cluster_num)]
    for i in range(len(cluster_labels)):
        cluster[cluster_labels[i]].append(label[i])

    cluster_to_y_label = []
    cluster_to_y_label_intra_cluster_num = []
    for i in range(cluster_num):
        temp = [0 for x in range(cluster_num)]
        for j in range(len(cluster[i])):
            temp[cluster[i][j]] = temp[cluster[i][j]] + 1
        cluster_to_y_label.append(np.argmax(temp))
        cluster_to_y_label_intra_cluster_num.append(temp[np.argmax(temp)])

    print(cluster_to_y_label)
    print(cluster_to_y_label_intra_cluster_num)
    temp = []
    for i in range(cluster_num):
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
    

def cifar10():
    
    num_classes = 10
    unseen_class = [0]
    num_unseen_classes = 1
    path = 'data/cifar10/'
    sum_unseen = 0
    sum_seen = 0

    x_test = np.load(path+'x_test.npy')
    y_test = np.load(path+'y_test.npy')
    unseen_x = np.load(path+'unseen_x.npy')
    unseen_y = np.load(path+'unseen_y.npy')
    
    unseen_x = unseen_x[:1000].astype('float32') / 255.
    unseen_y = unseen_y[:1000].reshape(1000)
    
    y_enc = load_model('model/cifar10/y_encoder.h5')
    mean_enc = load_model('model/cifar10/mean_encoder.h5')
    var_enc = load_model('model/cifar10/var_encoder.h5')

    ym = y_enc.predict(np.eye(num_classes))

    mean_seen = mean_enc.predict([x_test], batch_size=200)
    var_seen = var_enc.predict([x_test], batch_size=200)
    
    mean_unseen = mean_enc.predict([unseen_x], batch_size=200)
    var_unseen = var_enc.predict([unseen_x], batch_size=200)
    
    
    print('================print mean/var distribution=====================================')
    print('----------------seen----------------------------')
    print_dis(mean_seen, var_seen)
    print('----------------unseen----------------------------')
    print_dis(var_unseen, var_unseen)
    print()
    print()
    print()
    print('================y and posterior=====================================')
    sum_seen = y_posterior(y_test, ym, mean_seen, var_seen)
    sum_unseen = y_posterior(unseen_y, ym, mean_unseen, var_unseen)
    print('-----------------seen class correct num and rate-----------------')
    print(sum_seen, sum_seen/len(y_test))
    print('-----------------unseen class correct num and rate--------------')
    print(sum_unseen, sum_unseen/len(unseen_y))
    print('-----------------mix correct num and rate---------------------')
    print(sum_unseen + sum_seen, (sum_unseen + sum_seen) / (len(unseen_y) + len(y_test)))
    print()
    print()
    print()
    print('================y and cosine_similarity=====================================')
    sum_seen, thers = y_cosine_similarity(y_test, ym, mean_seen, var_seen)
    sum_unseen = y_cosine_similarity(unseen_y, ym, mean_unseen, var_unseen, thers)
    print('-----------------seen class correct num and rate-------------')
    print(sum_seen, sum_seen/len(y_test))
    print('-----------------unseen class correct num and rate-----------------')
    print(sum_unseen, sum_unseen/len(unseen_y))
    print('-----------------mix correct num and rate-------------------')
    print(sum_unseen + sum_seen, (sum_unseen + sum_seen) / (len(unseen_y) + len(y_test)))
    print()
    print()
    print()
    print('================kmeans=====================================')
    data = np.concatenate((mean_seen, mean_unseen))
    label = np.concatenate((y_test, unseen_y))
    print('-----------------test cluster num-------------------')
    test_cluster_num(data)
    print('-----------------kmeans-------------------')
    
    kmeans(data, label, 10)
    
cifar10()