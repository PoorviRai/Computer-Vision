# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:02:51 2020

@author: Poorvi Rai
"""

#Single Gaussian Distribution#
from DataHandling import load_training_data, load_test_data, get_MC, perform_pca, preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2

len_train = 1000
len_test = 100
size = 10

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def pdf(self, data):
        term1 = -0.5 * (data - self.mu).T
        term2 = np.linalg.inv(self.sigma)
        term3 = data - self.mu
        expo1 = np.matmul(term1, term2)
        expo2 = np.matmul(expo1, term3)[0,0]
        val =  np.exp(expo2)
        val =  val/np.sqrt(np.linalg.det(self.sigma))
        
        for i in range(0, size*size//2):
            val = val/(2*np.pi)
        
        return val

    
print("Loading Data")
X_train_face = load_training_data('face')
X_train_nonface = load_training_data('nonface')

[X_train_PCA_face, PCA_face_train] = perform_pca(X_train_face, size*size)
X_train_PCA_face = X_train_PCA_face.T
X_train_PCA_face = preprocess(X_train_PCA_face)

[X_train_PCA_nonface, PCA_nonface_train] = perform_pca(X_train_nonface, size*size)
X_train_PCA_nonface = X_train_PCA_nonface.T
X_train_PCA_nonface = preprocess(X_train_PCA_nonface)

[mean_face, covar_face] = get_MC(X_train_PCA_face)
mix_face = Gaussian(mean_face.reshape(-1,1), covar_face)
    
[mean_nonface, covar_nonface] = get_MC(X_train_PCA_nonface)
mix_nonface = Gaussian(mean_nonface.reshape(-1,1), covar_nonface)
  
   
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
    
    prob_face_facedata = np.append(prob_face_facedata, mix_face.pdf(inp_facedata))
    prob_nonface_facedata = np.append(prob_nonface_facedata, mix_nonface.pdf(inp_facedata))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, mix_face.pdf(inp_nonfacedata))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, mix_nonface.pdf(inp_nonfacedata))


post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata)
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata)
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata)


cov1 = mix_face.sigma
min_val = np.min(cov1)
max_val = np.max(cov1)
cov1 = ((cov1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
cv2.imshow('Cov1', cov1)

cov2 = mix_nonface.sigma
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


#Visualization
face_original = np.dot(mix_face.mu[:,0], PCA_face_train.components_) + PCA_face_train.mean_
face_original = np.array(face_original).astype('uint8')
face_mean = np.reshape(face_original,(60,60))

nonface_original = np.dot(mix_nonface.mu[:,0], PCA_nonface_train.components_) + PCA_nonface_train.mean_
nonface_original = np.array(nonface_original).astype('uint8')
nonface_mean = np.reshape(nonface_original,(60,60))

cv2.imshow("Mean Face", face_mean)
cv2.imshow("Mean Non-Face", nonface_mean)
cv2.waitKey(0)

cv2.destroyAllWindows()
