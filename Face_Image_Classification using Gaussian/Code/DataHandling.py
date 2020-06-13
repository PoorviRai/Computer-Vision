# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:02:51 2020

@author: Poorvi Rai
"""

import numpy as np
import cv2
import os
from sklearn import preprocessing
from sklearn import decomposition

len_train = 1000
len_test = 100
size = 10

train_face = 'train_face_images/'
train_nonface = 'train_nonface_images/'
test_face = 'test_face_images/'
test_nonface = 'test_nonface_images/'

files_train_face = os.listdir(train_face)
files_train_nonface = os.listdir(train_nonface)
files_test_face = os.listdir(test_face)
files_test_nonface = os.listdir(test_nonface)

def get_training_image(index, img_type):
    dat = []
    
    if img_type == 'face':
        dat = cv2.imread(train_face + files_train_face[index])
    elif img_type == 'nonface':
        dat = cv2.imread(train_nonface + files_train_nonface[index])
    
    dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')
    
    return dat
    

def get_test_image(index,img_type):
    dat = []

    if img_type == 'face':
        dat = cv2.imread(test_face + files_test_face[index])
    elif img_type == 'nonface':
        dat = cv2.imread(test_nonface + files_test_nonface[index])

    dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY).astype('float')

    return dat

   
def perform_pca(X, n_components):
    pca = decomposition.PCA(n_components = n_components)
    pca.fit(X)
    X_PCA = pca.transform(X)
    return X_PCA, pca   


def preprocess(X):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_PP = scaler.transform(X)
    return X_PP
 
    
def load_training_data(img_type):
    X_train = []
    
    for i in range(0, len_train):
        if img_type == 'face' :
            dat = get_training_image(i, 'face')
            dat = dat.flatten()
            X_train.append(dat)
        elif img_type == 'nonface' :        
            dat = get_training_image(i, 'nonface')
            dat = dat.flatten()
            X_train.append(dat)
    
    X_train = np.array(X_train)
    return X_train


def load_test_data(img_type):
    X_test = []
    
    for i in range(0, len_test):
        if img_type == 'face':
            dat = get_test_image(i, 'face')
            dat = dat.flatten()
            X_test.append(dat)
        elif img_type == 'nonface':
            dat = get_test_image(i, 'nonface')
            dat = dat.flatten()
            X_test.append(dat)
    
    X_test = np.array(X_test)        
    return X_test


def get_MC(X):
    meanX = np.mean(X, axis=1)
    covar = np.zeros((size*size, size*size)).astype('float64')
    covariance = np.cov(X)
    diag = covariance.diagonal()
    np.fill_diagonal(covar,diag)
    return [meanX, covar]
