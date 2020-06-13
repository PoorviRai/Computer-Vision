"""
Created on Mon Mar 23 09:15:20 2020

@author: Poorvi Rai
"""

from Config import TEST_FACE, TEST_NONFACE, TRAINING_IMG_HEIGHT, TRAINING_IMG_WIDTH, ADABOOST_CACHE_FILE, LABEL_POSITIVE
from AdaBoost import getCachedModel
from PreProcess import ImageSet
from HaarFeature import Feature

import numpy

face = ImageSet(TEST_FACE, sampleNum = 100)
nonFace = ImageSet(TEST_NONFACE, sampleNum = 100)

tot_samples = face.sampleNum + nonFace.sampleNum

haar = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

mat = numpy.zeros((haar.featuresNum, tot_samples))

for i in range(face.sampleNum):
    featureVec = haar.calFeatureForImg(face.images[i])
    for j in range(haar.featuresNum):
        mat[j][i] = featureVec[j]
        
for i in range(nonFace.sampleNum):
    featureVec = haar.calFeatureForImg(nonFace.images[i])
    for j in range(haar.featuresNum):
        mat[j][i + face.sampleNum] = featureVec[j]


model = getCachedModel(filename = ADABOOST_CACHE_FILE, limit = 10)

output = model.prediction(mat, th=0)

detectionRate = numpy.count_nonzero(output[0:100] == LABEL_POSITIVE) * 1.0 / 100

print (output)



