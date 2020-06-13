# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:27:45 2020

@author: Poorvi Rai
"""

#t-Distribution#
from DataHandling import load_training_data, load_test_data, get_MC, perform_pca, preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import optimize, special
import cv2

len_train = 1000
len_test = 100
size = 10

E_h = np.zeros(len_train)
E_log_h = np.zeros(len_train)
delta = np.zeros(len_train)        


def get_delta(X, i, u, sigma):
    term1 = np.matmul((X[:,i].reshape(-1,1) - u).T, np.linalg.inv(sigma))                                  
    term2 = np.matmul(term1, (X[:,i].reshape(-1,1) - u))
    return term2    
    

def prob(i, v, u, sigma, X):
    D = u.shape[0]
    c1 = special.gamma((v + D)/2.0) / (pow((v * np.pi), D/2) * np.sqrt(np.linalg.det(sigma)) * special.gamma(v/2))
    term2 = get_delta(X, i, u, sigma)    
    c2 = (1 + term2/v)
    val = c1 * pow(c2, -(v + D)/2)
    return val[0,0]


def get_E_h(i, v, u, sigma, X):
    D = u.shape[0]
    term1 = np.matmul((X[:,i].reshape(-1,1) - u).T, np.linalg.inv(sigma))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1) - u)
    val = (v + D)/(v + term2)
    return val
    

def get_E_log_h(i, v, u, sigma, X):
    D = u.shape[0]
    term1 = np.matmul((X[:,i].reshape(-1,1) - u).T, np.linalg.inv(sigma))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1) - u)
    val = special.digamma((v + D)/2) - np.log((v + term2)/2)
    return val


def tCost(v):
    val = 0
    
    for i in range(0,len_train):
       val = val + (v/2 - 1) * E_log_h[i] - (v/2) * E_h[i] - (v/2) * np.log(v/2) - np.log(special.gamma(v/2))                 
    
    return -val


def E_step(v, u, sigma, X):
    for i in range(0, len_train):
        term = np.matmul((X[:,i].reshape(-1,1) - u).T, np.linalg.inv(sigma))                                  
        delta[i] = np.matmul(term, (X[:,i].reshape(-1,1) - u))
        E_h[i] = get_E_h(i, v, u, sigma, X)
        E_log_h[i] = get_E_log_h(i, v, u, sigma, X)
    
    return [delta, E_h, E_log_h]
        

#update mean, variance and return argmin
def perform_EM_round(v, u, sigma, X):
    D = u.shape[0] 
    [delta, E_h, E_log_h] = E_step(v, u, sigma, X)
               
    temp_mean = np.zeros((D,1))
    num = np.zeros((D,D))
    denum = 0
    
    for i in range(0,len_train):
        temp_mean = temp_mean + E_h[i] * X[:,i].reshape(-1,1)
        denum = denum + E_h[i]
    
    u = temp_mean/denum    
        
    
    for i in range(0,len_train):
        prod = np.matmul((X[:,i].reshape(-1,1) - u), (X[:,i].reshape(-1,1) - u).T)
        num = num + E_h[i] * prod
    
    sigma = num/denum
    sigma = np.diag(np.diag(sigma))        


    v = optimize.fmin(tCost, v)          
     
    return [v[0], u, sigma]  


print("Loading data")
X_train_face = load_training_data('face')
X_train_nonface = load_training_data('nonface')

[X_train_PCA_face, PCA_face_train] = perform_pca(X_train_face, size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface, PCA_nonface_train] = perform_pca(X_train_nonface, size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)

[mean_face, covar_face] = get_MC(X_train_PCA_face)
[mean_nonface, covar_nonface] = get_MC(X_train_PCA_nonface)


X_test_face = load_test_data('face') 
X_test_nonface = load_test_data('nonface') 
          
[X_test_face_PCA, PCA_face_test] = perform_pca(X_test_face, size*size)
X_test_face_PCA = X_test_face_PCA.T
X_test_face_PCA = preprocess(X_test_face_PCA)
              
[X_test_nonface_PCA, PCA_nonface_test] = perform_pca(X_test_nonface, size*size)
X_test_nonface_PCA = X_test_nonface_PCA.T
X_test_nonface_PCA = preprocess(X_test_nonface_PCA)


v_face = 50
u_face = mean_face
sigma_face = covar_face

v_nonface = 50
u_nonface = mean_nonface
sigma_nonface = covar_nonface

len_train = 1000
n_iter = 20

print("Training")
for i in range(0, n_iter):
    print("")
    print(i)
    [v_face, u_face, sigma_face] = perform_EM_round(v_face, u_face.reshape(-1,1), sigma_face, X_train_PCA_face)
    print ("v_face = ", v_face) 

for i in range(0, n_iter):
    print("")
    print(i)
    [v_nonface, u_nonface, sigma_nonface] = perform_EM_round(v_nonface, u_nonface.reshape(-1,1), sigma_nonface, X_train_PCA_nonface)
    print("v_nonface = ", v_nonface)


print("Testing")
prob_face_facedata = np.array([])
prob_nonface_facedata = np.array([])
prob_face_nonfacedata = np.array([])
prob_nonface_nonfacedata = np.array([])

for i in range(0, len_test):
    prob_face_facedata = np.append(prob_face_facedata, prob(i, v_face, u_face.reshape(-1,1), sigma_face, X_train_PCA_face))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, prob(i, v_face, u_face.reshape(-1,1), sigma_face, X_train_PCA_nonface))    
    prob_nonface_facedata = np.append(prob_nonface_facedata, prob(i, v_nonface, u_nonface.reshape(-1,1), sigma_nonface, X_train_PCA_face))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, prob(i, v_nonface, u_nonface.reshape(-1,1), sigma_nonface, X_train_PCA_nonface))
        
    
post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata)
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata)
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)


cov1 = sigma_face
min_val = np.min(cov1)
max_val = np.max(cov1)
cov1 = ((cov1 - min_val)/(max_val - min_val) * 255.0).astype('uint8')
cv2.imshow('Cov1', cov1)

cov2 = sigma_nonface
min_val = np.min(cov2)
max_val = np.max(cov2)
cov2 = ((cov2 - min_val)/(max_val - min_val) * 255.0).astype('uint8')
cv2.imshow('Cov2', cov2)
cv2.waitKey(0)


print("Plotting ROC")
predictions = np.append(post_face_nonface_data, post_face_face_data)
temp1 = [0] * len_test
temp2 = [1] * len_test
actual = np.append(temp1, temp2)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


#Visualization
face_original = np.dot(u_face[:,0], PCA_face_train.components_) + PCA_face_train.mean_
face_original = np.array(face_original).astype('uint8')
face_mean = np.reshape(face_original, (60,60))

nonface_original = np.dot(u_nonface[:,0], PCA_nonface_train.components_) + PCA_nonface_train.mean_
nonface_original = np.array(nonface_original).astype('uint8')
nonface_mean = np.reshape(nonface_original, (60,60))

cv2.imshow("Mean Face",face_mean)
cv2.imshow("Mean Non-Face",nonface_mean)
cv2.waitKey(0)

cv2.destroyAllWindows()
