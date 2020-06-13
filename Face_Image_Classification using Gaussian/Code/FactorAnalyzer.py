# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:33:19 2020

@author: Poorvi Rai
"""

from DataHandling import load_training_data, load_test_data, get_MC, perform_pca, preprocess
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2

len_train = 1000
len_test = 100
size = 10
K = 5

class factor:
    def __init__(self, u, sigma, phi):
        self.u = u
        self.sigma = sigma
        self.phi = phi
        self.E_h = np.zeros((len_train, K, size*size))
        self.E_hi_hT = np.zeros((len_train, K, K))


    def prob(self, X, i):
        sigma = np.matmul(self.phi, self.phi.T) + self.sigma
        term1 = -0.5 * (X[:,i].reshape(-1,1) - self.u).T
        term2 = np.linalg.inv(sigma)
        term3 = X[:,i].reshape(-1,1) - self.u
        expo1 = np.matmul(term1,term2)
        expo2 = np.matmul(expo1,term3)[0,0]
        val =  np.exp(expo2)
        val =  val/np.sqrt(np.linalg.det(sigma))
        
        for i in range(0, size*size//2):
            val = val/(2 * np.pi)
        
        return val


    #update E_h and E_hi_hT
    def E_step(self,X):
        for i in range(0, len_train):
            t1 = np.matmul(self.phi.T, np.linalg.inv(self.sigma))
            t2 = np.matmul(t1, self.phi) + np.eye(K)
            t3 = np.matmul(np.linalg.inv(t2), self.phi.T)
            t4 = np.matmul(t3, np.linalg.inv(self.sigma))
            
            self.E_h[i] = np.matmul(t4, X[:,i] - self.u)
            self.E_hi_hT[i] = np.linalg.inv(t2) + np.matmul(self.E_h[i], self.E_h[i].T)


    #update phi and sigma
    def M_step(self, X):
        t1 = np.zeros((size*size, K))
        t2 = np.zeros((K, K))
        temp = np.zeros((size*size,size*size))
        
        for i in range(0, len_train):
            t1 = t1 + np.matmul((X[:,i] - self.u) , self.E_h[i].T) 
            t2 = t2 + self.E_hi_hT[i]
        
        self.phi = np.matmul(t1, t2)    
                
        for i in range(0, len_train):
            temp = temp + np.matmul(X[:,i].reshape(-1,1) - self.u, (X[:,i].reshape(-1,1) - self.u).T)
            t2 = np.matmul(self.phi, self.E_h[i])
            t3 = np.matmul(t2, (X[:,i].reshape(-1,1) - self.u ))
            temp = temp - t3
        
        temp = temp/len_train
        self.sigma = temp
        self.sigma = np.diag(np.diag(temp))


    def perform_EM_step(self, X):
        self.E_step(X)
        self.M_step(X)        
        

print("Loading data")       
X_train_face = load_training_data('face')
X_train_nonface = load_training_data('nonface')

[X_train_PCA_face, PCA_face_train] = perform_pca(X_train_face, size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface, PCA_nonface_train] = perform_pca(X_train_nonface, size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)

[mean_face,covar_face] = get_MC(X_train_PCA_face)
[mean_nonface,covar_nonface] = get_MC(X_train_PCA_nonface)


X_test_face = load_test_data('face')
X_test_nonface = load_test_data('nonface')
           
[X_test_face_PCA, PCA_face_test] = perform_pca(X_test_face, size*size)
X_test_face_PCA = X_test_face_PCA.T
X_test_face_PCA = preprocess(X_test_face_PCA)
               
[X_test_nonface_PCA, PCA_nonface_test] = perform_pca(X_test_nonface, size*size)
X_test_nonface_PCA = X_test_nonface_PCA.T
X_test_nonface_PCA = preprocess(X_test_nonface_PCA)


phi_face = np.random.rand(size*size, K)
sigma_face = np.random.rand(size*size, size*size)
sigma_face = np.diag(np.diag(sigma_face))

phi_nonface = np.random.rand(size*size, K)
sigma_nonface = np.random.rand(size*size, size*size)
sigma_nonface = np.diag(np.diag(sigma_nonface))

mix_face = factor(mean_face.reshape(-1,1), sigma_face, phi_face)
mix_nonface = factor(mean_nonface.reshape(-1,1), sigma_nonface, phi_nonface)


[X_train_PCA_face,PCA_face_train] = perform_pca(X_train_face,size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface,PCA_nonface_train] = perform_pca(X_train_nonface,size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)

[mean_face,covar_face] = get_MC(X_train_PCA_face)
[mean_nonface,covar_nonface] = get_MC(X_train_PCA_nonface)


for i in range(0, 10):
    mix_face.perform_EM_step(X_train_PCA_face)
    mix_nonface.perform_EM_step(X_train_PCA_nonface)
      

print("Testing")
prob_face_facedata = np.array([])
prob_nonface_facedata = np.array([])
prob_face_nonfacedata = np.array([])
prob_nonface_nonfacedata = np.array([])

for i in range(0, len_test):
    prob_face_facedata = np.append(prob_face_facedata, mix_face.prob(X_test_face_PCA, i))
    prob_nonface_facedata = np.append(prob_nonface_facedata, mix_nonface.prob(X_test_face_PCA, i))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, mix_face.prob(X_test_nonface_PCA, i))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, mix_nonface.prob(X_test_nonface_PCA, i))


post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata)
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata)
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)


print("Plotting ROC")
predictions = np.append( post_face_nonface_data , post_face_face_data )
temp1 = [0]*len_test
temp2 = [1]*len_test
actual = np.append(temp1,temp2)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


#Visualization
face_original = np.dot(mean_face, PCA_face_train.components_) + PCA_face_train.mean_
face_original = np.array(face_original).astype('uint8')
face_mean = np.reshape(face_original,(60,60))

nonface_original = np.dot(mean_nonface, PCA_nonface_train.components_) + PCA_nonface_train.mean_
nonface_original = np.array(nonface_original).astype('uint8')
nonface_mean = np.reshape(nonface_original,(60,60))

cv2.imshow("Mean Face", face_mean)
cv2.imshow("Mean Non-Face", nonface_mean)
cv2.waitKey(0)

cv2.destroyAllWindows()
