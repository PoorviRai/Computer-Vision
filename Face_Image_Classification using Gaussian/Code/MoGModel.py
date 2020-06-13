# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:14:22 2020

@author: Poorvi Rai
"""

#Gaussian Mixture#
from DataHandling import load_training_data, load_test_data, perform_pca, preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2

len_train = 1000
len_test = 100
size = 10
K = 3
np.random.seed(3)

class GaussianMix():
    def __init__(self, u, sigma):
        self.u = u
        self.sigma = sigma
        self.length = self.u.shape[1]        
        self.lambda_val = np.random.dirichlet(np.ones(K), size = 1)[0]
        self.r = np.random.dirichlet(np.ones(K), size = len_train)
        
        
    def pdf(self, k_index, i_index, X):
        term1 = -0.5 * (X[:,i_index].reshape(-1,1) - self.u[k_index]).T
        term2 = np.linalg.inv(self.sigma[k_index])
        term3 = X[:,i_index].reshape(-1,1) - self.u[k_index]
        expo1 = np.matmul(term1, term2)
        expo2 = np.matmul(expo1, term3)[0,0]
        val =  np.exp(expo2)
        val =  val/np.sqrt(np.linalg.det(self.sigma[k_index]))
        
        for i in range(0, size*size//2):
            val = val/(2*np.pi)
        
        return val


    def get_prob(self, i_index, X):
        val = 0
        
        for k_index in range(0,K):
            val = val + self.lambda_val[k_index] * self.pdf(k_index, i_index, X)
        
        return val         
        
    
    #update matrix
    def E_step(self,i_index, k_index, X):   
        temp = 0
        
        for j in range(0, K):
            temp = temp + self.lambda_val[j] * self.pdf(j, i_index, X)
        
        self.r[i_index,k_index] = self.lambda_val[k_index] * self.pdf(k_index, i_index, X)/temp   
    
    
    #update lambda
    def M_step(self, i_index, k_index):
        num = 0
        denum = 0
        
        for i in range(0, len_train):
            num = num + self.r[i, k_index]
            
            for j in range(0, K):
                denum = denum + self.r[i,j]      
        
        self.lambda_val[k_index] = 1.0 * num/denum
        
      
    #update u and sigma          
    def update(self, i_index, k_index, X): 
        num = np.zeros((self.length, 1))
        denum = 0
        
        for i in range(0, len_train):
            num = num + self.r[i, k_index] * X[:,i].reshape(-1,1)
            denum = denum + self.r[i, k_index]    
        
        self.u[k_index] = 1.0 * num/denum
        
        num = np.zeros((self.length, self.length))
        denum = 0
        
        for i in range(0, len_train):
            num = num + self.r[i, k_index] * np.matmul((X[:,i].reshape(-1,1) - self.u[k_index]), (X[:,i].reshape(-1,1) - self.u[k_index]).T)
            denum = denum + self.r[i, k_index]
        
        self.sigma[k_index] = 1.0 * num/denum    
        self.sigma[k_index] = np.diag(np.diag(self.sigma[k_index]))
    
    
    #update all
    def perform_EM(self, k_index, X):
        for i in range(0, len_train):
            self.E_step(i, k_index, X)
            self.M_step(i, k_index)
            self.update(i, k_index, X)
               

print("Loading Training data")
X_train_face = load_training_data('face')
X_train_nonface = load_training_data('nonface')

[X_train_PCA_face, PCA_face_train] = perform_pca(X_train_face, size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface, PCA_nonface_train] = perform_pca(X_train_nonface, size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)

mean_face = np.zeros((3,100,1))
mean_nonface = np.zeros((3,100,1))

covar_face = np.zeros((3,100,100))
np.fill_diagonal(covar_face[0], np.random.rand(100))
np.fill_diagonal(covar_face[1], np.random.rand(100))
np.fill_diagonal(covar_face[2], np.random.rand(100))

covar_nonface = np.zeros((3,100,100))
np.fill_diagonal(covar_nonface[0], np.random.rand(100))
np.fill_diagonal(covar_nonface[1], np.random.rand(100))
np.fill_diagonal(covar_nonface[2], np.random.rand(100))

mix_face = GaussianMix(mean_face, covar_face)
mix_nonface = GaussianMix(mean_nonface, covar_nonface)


print("Performing EM Face")
for k in range(0, K):
    print("Face Component - ", k)
    mix_face.perform_EM(k, X_train_PCA_face)
    print("mix_face -> ", mix_face.lambda_val)


print("Performing EM Non-Face")
for k in range(0,K):    
    print("Non-Face Component - ", k)
    mix_nonface.perform_EM(k, X_train_PCA_nonface)    
    print("mix_nonface -> ", mix_nonface.lambda_val)
   


print("Loading Test data")
X_test_face = load_test_data('face')
X_test_nonface = load_test_data('nonface')
           
[X_test_face_PCA, PCA_face_test] = perform_pca(X_test_face, size*size)
X_test_face_PCA = X_test_face_PCA.T
X_test_face_PCA = preprocess(X_test_face_PCA)
               
[X_test_nonface_PCA, PCA_nonface_test] = perform_pca(X_test_nonface, size*size)
X_test_nonface_PCA = X_test_nonface_PCA.T
X_test_nonface_PCA = preprocess(X_test_nonface_PCA)


print("Testing")
prob_face_facedata = np.array([])
prob_nonface_facedata = np.array([])
prob_face_nonfacedata = np.array([])
prob_nonface_nonfacedata = np.array([])

for i in range(0, len_test):
    inp_facedata = X_test_face_PCA[:,i].reshape(-1,1)
    inp_nonfacedata = X_test_nonface_PCA[:,i].reshape(-1,1)
    
    prob_face_facedata = np.append(prob_face_facedata, mix_face.get_prob(i, X_test_face_PCA))
    prob_nonface_facedata = np.append(prob_nonface_facedata, mix_nonface.get_prob(i, X_test_face_PCA))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, mix_face.get_prob(i, X_test_nonface_PCA))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, mix_nonface.get_prob(i, X_test_nonface_PCA))


post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata)
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata)
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)


cov1 = mix_face.lambda_val[0] * mix_face.sigma[0] + mix_face.lambda_val[1] * mix_face.sigma[1] + mix_face.lambda_val[2] * mix_face.sigma[2] 
min_val = np.min(cov1)
max_val = np.max(cov1)
cov1 = ((cov1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
cv2.imshow('Cov1', cov1)

cov2 = mix_nonface.lambda_val[0] * mix_nonface.sigma[0] + mix_nonface.lambda_val[1] * mix_nonface.sigma[1] + mix_nonface.lambda_val[2] * mix_nonface.sigma[2] 
min_val = np.min(cov2)
max_val = np.max(cov2)
cov2 = ((cov2-min_val)/(max_val-min_val) * 255.0).astype('uint8')
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
plt.xlabel('False Rositive Rate')
plt.ylabel('True Positive Rate')


#Visualization
m0 = mix_face.u[0]
m1 = mix_face.u[1]
m2 = mix_face.u[2]

face_0 = np.dot(m0[:,0], PCA_face_train.components_) + PCA_face_train.mean_
face_1 = np.dot(m1[:,0], PCA_face_train.components_) + PCA_face_train.mean_
face_2 = np.dot(m2[:,0], PCA_face_train.components_) + PCA_face_train.mean_

face_0 = ((face_0 - np.min(face_0))/(np.max(face_0) - np.min(face_0)) * 255.0).astype('uint8')
face_1 = ((face_1 - np.min(face_1))/(np.max(face_1) - np.min(face_1)) * 255.0).astype('uint8')
face_2 = ((face_2 - np.min(face_2))/(np.max(face_2) - np.min(face_2)) * 255.0).astype('uint8')

face_0 = np.reshape(face_0, (60,60))
face_1 = np.reshape(face_1, (60,60))
face_2 = np.reshape(face_2, (60,60))

cv2.imshow("face0", face_0)
cv2.imshow("face1", face_1)
cv2.imshow("face2", face_2)
cv2.waitKey(0)


m0 = mix_nonface.u[0]
m1 = mix_nonface.u[1]
m2 = mix_nonface.u[2]

nonface_0 = np.dot(m0[:,0], PCA_nonface_train.components_) + PCA_nonface_train.mean_
nonface_1 = np.dot(m1[:,0], PCA_nonface_train.components_) + PCA_nonface_train.mean_
nonface_2 = np.dot(m2[:,0], PCA_nonface_train.components_) + PCA_nonface_train.mean_

nonface_0 = ((nonface_0 - np.min(nonface_0))/(np.max(nonface_0) - np.min(nonface_0)) * 255.0).astype('uint8')
nonface_1 = ((nonface_1 - np.min(nonface_1))/(np.max(nonface_1) - np.min(nonface_1)) * 255.0).astype('uint8')
nonface_2 = ((nonface_2 - np.min(nonface_2))/(np.max(nonface_2) - np.min(nonface_2)) * 255.0).astype('uint8')

nonface_0 = np.reshape(nonface_0, (60,60))
nonface_1 = np.reshape(nonface_1, (60,60))
nonface_2 = np.reshape(nonface_2, (60,60))

cv2.imshow("nonface0", nonface_0)
cv2.imshow("nonface1", nonface_1)
cv2.imshow("nonface2", nonface_2)
cv2.waitKey(0)

cv2.destroyAllWindows()
