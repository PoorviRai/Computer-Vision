# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:35:18 2020

@author: Poorvi Rai
"""

from DataHandling import load_training_data, load_test_data, perform_pca, preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import optimize, special
import cv2

len_train = 1000
len_test = 100
size = 10
K = 3

E_h = np.zeros((K, len_train))
E_log_h = np.zeros((K, len_train))
delta = np.zeros((K, len_train))      


def get_delta(i, k, u, sigma, X):
    term1 = np.matmul((X[:,i].reshape(-1,1) - u[k]).T, np.linalg.inv(sigma[k])) 
    term2 = np.matmul(term1, (X[:,i].reshape(-1,1) - u[k]))
    return term2    
  
    
def prob(i, k, v, u, sigma, X):
    D = u[k].shape[0]
    c1 = special.gamma((v[k] + D)/2.0)/(pow((v[k] * np.pi), D/2) * np.sqrt(np.linalg.det(sigma[k])) * special.gamma(v[k]/2))
    term2 = get_delta(i, k, u, sigma, X)    
    c2 = (1 + term2/v[k])
    val = c1 * pow(c2, -(v[k] + D)/2)
    return val[0,0] 


def get_prob(i, v, u, sigma, X):
    val = 0
    
    for k in range(0, K):
        val = val + prob(i, k, v, u, sigma, X)
    
    return val
    

def get_E_h(i, k, v, u, sigma, X):
    D = u.shape[1]
    term1 = np.matmul((X[:,i].reshape(-1,1) - u[k]).T, np.linalg.inv(sigma[k]))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1) - u[k])
    val = (v[k] + D)/(v[k] + term2)
    return val
    

def get_E_log_h(i, k, v, u, sigma, X):
    D = u.shape[1]
    term1 = np.matmul((X[:,i].reshape(-1,1) - u[k]).T, np.linalg.inv(sigma[k]))
    term2 = np.matmul(term1, X[:,i].reshape(-1,1) - u[k])
    val = special.digamma((v[k] + D)/2) - np.log((v[k] + term2)/2)
    return val


def tCost0(v):
    val = 0
    
    for i in range(0, len_train):
       val = val + ((v[0]/2) - 1) * E_log_h[0,i] - (v[0]/2) * E_h[0,i] - (v[0]/2) * np.log(v[0]/2) - np.log(special.gamma(v[0]/2))        
    
    return -val


def tCost1(v):
    val = 0
    
    for i in range(0, len_train):
       val = val + ((v[1]/2) - 1) * E_log_h[1,i] - (v[1]/2) * E_h[1,i] - (v[1]/2) * np.log(v[1]/2) - np.log(special.gamma(v[1]/2))                 
    
    return -val


def tCost2(v):
    val = 0
    
    for i in range(0,len_train):
       val = val + ((v[2]/2) - 1) * E_log_h[2,i] - (v[2]/2) * E_h[2,i] - (v[2]/2) * np.log(v[2]/2) - np.log(special.gamma(v[2]/2))                 
    
    return -val


def E_step(k, v, u, sigma, X):
    for i in range(0, len_train):
        term = np.matmul((X[:,i].reshape(-1,1) - u[k]).T, np.linalg.inv(sigma[k]))
        delta[k,i] = np.matmul(term, (X[:,i].reshape(-1,1) - u[k]))
        E_h[k,i] = get_E_h(i, k, v, u, sigma, X)
        E_log_h[k,i] = get_E_log_h(i, k, v, u, sigma, X)
    
    return [delta, E_h, E_log_h]
        

#update mean, variance and return argmin
def perform_EM_round(k, v, u, sigma, X):
    D = u.shape[1]
    [delta, E_h, E_log_h] = E_step(k, v, u, sigma, X)
               
    temp_mean = np.zeros((D,1))
    num = np.zeros((D,D))
    denum = 0
    
    for i in range(0, len_train):
        temp_mean = temp_mean + E_h[k,i] * X[:,i].reshape(-1,1)
        denum = denum + E_h[k,i]
    
    u[k] = temp_mean/denum    
    
    
    for i in range(0, len_train):
        prod = np.matmul((X[:,i].reshape(-1,1) - u[k]), (X[:,i].reshape(-1,1) - u[k]).T)
        num = num + E_h[k,i] * prod
    
    sigma[k] = num/denum
    sigma[k] = np.diag(np.diag(sigma[k]))


    if k == 0:
       v[k] = optimize.fmin(tCost0,[50,50,50])[0]            
    if k == 1:
       v[k] = optimize.fmin(tCost1,[50,50,50])[0]      
    if k == 2:
       v[k] = optimize.fmin(tCost2,[50,50,50])[0]      
    
    return [v, u, sigma]  


def perform_EM(v, u, sigma, X):
    for k in range(0, K):
        print("Component ", k)
        [v, u, sigma] = perform_EM_round(k, v, u, sigma, X)
    
    return [v,u,sigma]    


print("Loading data")  
X_train_face = load_training_data('face')
X_train_nonface = load_training_data('nonface')

[X_train_PCA_face, PCA_face_train] = perform_pca(X_train_face, size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface, PCA_nonface_train] = perform_pca(X_train_nonface, size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)


X_test_face = load_test_data('face')
X_test_nonface = load_test_data('nonface') 
         
[X_test_face_PCA, PCA_face_test] = perform_pca(X_test_face, size*size)
X_test_face_PCA = X_test_face_PCA.T
X_test_face_PCA = preprocess(X_test_face_PCA)
              
[X_test_nonface_PCA, PCA_nonface_test] = perform_pca(X_test_nonface, size*size)
X_test_nonface_PCA = X_test_nonface_PCA.T
X_test_nonface_PCA = preprocess(X_test_nonface_PCA)


v_face = [50,50,50]
u_face = np.random.rand(K, size**2, 1)
sigma_face = np.random.rand(K, size**2, size**2)

v_nonface = [50,50,50]
u_nonface = np.random.rand(K, size**2, 1)
sigma_nonface = np.random.rand(K, size**2, size**2)


print("Training")
n_iter = 20
k = 0

print("mix_face")
for i in range(0, n_iter):
    print("")
    print(i)
    [v_face, u_face, sigma_face] = perform_EM(v_face, u_face, sigma_face, X_train_PCA_face)
    print("v_face = ", v_face) 

print("mix_nonface")
for i in range(0, n_iter):
    print("")
    print(i)
    [v_nonface, u_nonface, sigma_nonface] = perform_EM(v_nonface, u_nonface, sigma_nonface, X_train_PCA_nonface)
    print ("v_nonface = ", v_nonface)
 

print("Testing")
prob_face_facedata = np.array([])
prob_nonface_facedata = np.array([])
prob_face_nonfacedata = np.array([])
prob_nonface_nonfacedata = np.array([])

for i in range(0, len_test):
    prob_face_facedata = np.append(prob_face_facedata, get_prob(i, v_face, u_face, sigma_face, X_train_PCA_face))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, get_prob(i, v_face, u_face, sigma_face, X_train_PCA_nonface))    
    prob_nonface_facedata = np.append(prob_nonface_facedata, get_prob(i, v_nonface, u_nonface, sigma_nonface, X_train_PCA_face))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, get_prob(i, v_nonface, u_nonface, sigma_nonface, X_train_PCA_nonface))


post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata)
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata)
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)


print("Plotting ROC")
predictions = np.append(post_face_nonface_data, post_face_face_data)
temp1 = [0] * len_test
temp2 = [1] * len_test
actual = np.append(temp1,temp2)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

