"""
Created on Sun Mar 22 01:13:58 2020

@author: Poorvi Rai
"""

from Config import POSITIVE_SAMPLE, NEGATIVE_SAMPLE, TRAINING_IMG_HEIGHT, TRAINING_IMG_WIDTH, FEATURE_NUM
from Config import ADABOOST_LIMIT, ADABOOST_CACHE_FILE, TRAINING_FACE, TRAINING_NONFACE
from AdaBoost import AdaBoost, getCachedModel
from HaarFeature import Feature
from PreProcess import ImageSet

import os
import numpy

Face = ImageSet(TRAINING_FACE, sampleNum = POSITIVE_SAMPLE)
nonFace = ImageSet(TRAINING_NONFACE, sampleNum = NEGATIVE_SAMPLE)

haar = Feature(TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT)

tot_samples = Face.sampleNum + nonFace.sampleNum

_mat = numpy.zeros((haar.featuresNum, tot_samples))

for i in range(Face.sampleNum):
    featureVec = haar.calFeatureForImg(Face.images[i])
    for j in range(haar.featuresNum):
        _mat[j][i] = featureVec[j]

for i in range(nonFace.sampleNum):
    featureVec = haar.calFeatureForImg(nonFace.images[i])
    for j in range(haar.featuresNum):
        _mat[j][i + Face.sampleNum] = featureVec[j]   

mat = _mat
featureNum, sampleNum = _mat.shape

assert sampleNum == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)
assert featureNum == FEATURE_NUM

Label_Face = [+1 for i in range(POSITIVE_SAMPLE)]
Label_NonFace = [-1 for i in range(NEGATIVE_SAMPLE)]

label = numpy.array(Label_Face + Label_NonFace)

cache_filename = ADABOOST_CACHE_FILE

if os.path.isfile(cache_filename):
    model = getCachedModel(mat = _mat, label = label, filename= cache_filename, limit = ADABOOST_LIMIT)
else:
    model = AdaBoost(mat, label, limit = ADABOOST_LIMIT)
    model.train()
    model.saveModel(cache_filename)

print (model)
