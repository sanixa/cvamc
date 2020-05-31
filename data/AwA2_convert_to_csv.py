#! -*- coding: utf-8 -*-



import pandas as pd
import os
import numpy as np
import cv2
import csv
from PIL import Image

image_size = 224          # 指定图片大小
path = '/Users/keiko/tensorflow/cvamc-master/data/AwA2/' 

seen_x = np.load(path + 'AwA2_seen_x_100.npy')
with open('AwA2_seen_x_train_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(3200):
    # 寫入一列資料
        writer.writerow(seen_x[i].reshape(image_size*image_size*3))
        
with open('AwA2_seen_x_test_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(3200,4000):
    # 寫入一列資料
        writer.writerow(seen_x[i].reshape(image_size*image_size*3))

seen_y = np.load(path + 'AwA2_seen_y_100.npy')
with open('AwA2_seen_y_train_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(3200):
    # 寫入一列資料
        writer.writerow([seen_y[i]])
        
with open('AwA2_seen_y_test_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(3200,4000):
    # 寫入一列資料
        writer.writerow([seen_y[i]])
'''
#########---------------------------------------------------------------------
unseen_x = np.load(path + 'AwA2_unseen_x_100.npy')
#print(train_x[0].reshape(image_size*image_size*3))

with open('AwA2_unseen_x_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(unseen_x)):
    # 寫入一列資料
        writer.writerow(unseen_x[i].reshape(image_size*image_size*3))

unseen_y = np.load(path + 'AwA2_unseen_y_100.npy')
#print(train_y[0])
with open('AwA2_unseen_y_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(len(unseen_y)):
    # 寫入一列資料
        writer.writerow([unseen_y[i]])


num_classes = 50
temp = np.eye(num_classes).reshape(num_classes*num_classes)

with open('AwA2_y_class_train_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(3200):
    # 寫入一列資料
        writer.writerow(temp)
        
with open('AwA2_y_class_test_100.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    for i in range(800):
    # 寫入一列資料
        writer.writerow(temp)

#y_class_train = np.repeat(a=temp, repeats=3200, axis=0).reshape(3200, num_classes, num_classes)
#y_class_test =  np.repeat(a=temp, repeats=800, axis=0).reshape(800, num_classes, num_classes)
'''