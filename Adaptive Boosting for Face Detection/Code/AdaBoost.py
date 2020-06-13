"""
Created on Sat Mar 21 16:41:39 2020

@author: Poorvi Rai
"""

from Config import LABEL_POSITIVE, LABEL_NEGATIVE, EXPECTED_TPR, EXPECTED_FPR, ROC_FILE
from WeakClassifier import WeakClassifier

from matplotlib import pyplot

import numpy
import time


def getCachedModel(mat = None, label = None, filename = "", limit = 0):
    fileObj = open(filename, "a+")

    print ("Constructing AdaBoost from existed model data")

    tmp = fileObj.readlines()

    if len(tmp) == 0:
        raise ValueError("There is no cached AdaBoost model")

    weakerNum = len(tmp) / 4
    model = AdaBoost(train = False, limit = weakerNum)

    if limit < weakerNum:
        model.weakerLimit = limit
    else:
        model.weakerLimit = weakerNum

    for i in range(0, len(tmp), 4):
        alpha, dimension, direction, threshold = None, None, None, None

        for j in range(i, i + 4):
            if   (j % 4) == 0:
                alpha = float(tmp[j])
            elif (j % 4) == 1:
                dimension = int(tmp[j])
            elif (j % 4) == 2:
                direction = float(tmp[j])
            elif (j % 4) == 3:
                threshold = float(tmp[j])

        classifier = model.Weaker(train = False)
        classifier.constructor(dimension, direction, threshold)
        classifier._mat = mat
        classifier._label = label

        if mat is not None:
            classifier.sampleNum = mat.shape[1]

        model.G[i/4] = classifier
        model.alpha[i/4] = alpha
        model.N += 1

    model._mat = mat
    model._label = label
    
    if model.N > limit:
        model.N    = limit

    if label is not None:
        model.samplesNum = len(label)

    print ("Construction finished")
    fileObj.close()

    return model


class AdaBoost:

    def __init__(self, Mat = None, Tag = None, classifier = WeakClassifier, train = True, limit = 4):
        if train == True:
            self._mat = Mat
            self._label = Tag

            self.samplesDim, self.samplesNum = self._mat.shape

            assert self.samplesNum == self._label.size

            self.posNum = numpy.count_nonzero(self._label == LABEL_POSITIVE)
            self.negNum = numpy.count_nonzero(self._label == LABEL_NEGATIVE)

            pos_W = [1.0/(2 * self.posNum) for i in range(self.posNum)]
            neg_W = [1.0/(2 * self.negNum) for i in range(self.negNum)]

            self.W = numpy.array(pos_W + neg_W)
            self.accuracy = []

        self.Weaker = classifier
        self.weakerLimit = limit
        self.G = [None for _ in range(limit)]
        self.alpha = [0  for _ in range(limit)]
        self.N = 0
        self.detectionRate = 0.

        # true positive rate
        self.tpr = 0.
        # false positive rate
        self.fpr = 0.
        #threshold
        self.th  = 0.


    def weakClassifierGood(self):

        output = self.prediction(self._mat, self.th)
        correct = numpy.count_nonzero(output == self._label)/(self.samplesNum * 1.0)
        
        self.accuracy.append(correct)
        self.detectionRate = numpy.count_nonzero(output[0:self.posNum] == LABEL_POSITIVE) * 1.0 / self.posNum

        Num_tp = 0 # Number of true positive
        Num_fn = 0 # Number of false negative
        Num_tn = 0 # Number of true negative
        Num_fp = 0 # Number of false positive
        
        for i in range(self.samplesNum):
            if self._label[i] == LABEL_POSITIVE:
                if output[i] == LABEL_POSITIVE:
                    Num_tp += 1
                else:
                    Num_fn += 1
            else:
                if output[i] == LABEL_POSITIVE:
                    Num_fp += 1
                else:
                    Num_tn += 1

        self.tpr = Num_tp * 1.0 / (Num_tp + Num_fn)
        self.fpr = Num_fp * 1.0 / (Num_tn + Num_fp)

        if self.tpr > EXPECTED_TPR and self.fpr < EXPECTED_FPR:
            return True


    def train(self):
        adaboost_start_time = time.time()

        for m in range(self.weakerLimit):
            self.N += 1

            weaker_start_time = time.time()

            self.G[m] = self.Weaker(self._mat, self._label, self.W)
            
            errorRate = self.G[m].train()

            print("Time for training WeakClassifier:", time.time() - weaker_start_time)

            if errorRate < 0.0001:
                errorRate = 0.0001

            beta = errorRate / (1 - errorRate)
            
            self.alpha[m] = numpy.log(1 / beta)

            output = self.G[m].prediction(self._mat)

            for i in range(self.samplesNum):
                if self._label[i] == output[i]:
                    self.W[i] *=  beta

            self.W /= sum(self.W)

            if self.weakClassifierGood():
                print ((self.N) ," weak classifier is enough to meet the request.")
                print ("Training Done")
                break

            print ("weakClassifier:", self.N)
            print ("errorRate     :", errorRate)
            print ("accuracy      :", self.accuracy[-1])
            print ("detectionRate :", self.detectionRate)
            print ("threshold     :", self.th)
            print ("alpha         :", self.alpha[m])

        self.showErrRates()
        self.showROC()

        print("The time cost of training this AdaBoost model:", time.time() - adaboost_start_time)

        output = self.prediction(self._mat, self.th)
        
        return output, self.fpr


    def grade(self, Mat):
        sampleNum = Mat.shape[1]
        output = numpy.zeros(sampleNum, dtype = numpy.float16)

        for i in range(self.N):
            output += self.G[i].prediction(Mat) * self.alpha[i]

        return output


    def prediction(self, Mat, th = None):
        output = self.grade(Mat)
            
        if th == None:
            th = self.th

        for i in range(len(output)):
            if output[i] > th:
                output[i] = LABEL_POSITIVE
            else:
                output[i] = LABEL_NEGATIVE

        return output


    def showErrRates(self):
        pyplot.title("The changes of accuracy")
        pyplot.xlabel("Iteration times")
        pyplot.ylabel("Accuracy of Prediction")
        
        pyplot.plot([i for i in range(self.N)], self.accuracy, '-.', label = "Accuracy * 100%")
        pyplot.axis([0., self.N, 0, 1.])

        pyplot.show()
        
        
    def showROC(self):
        low_bound = -sum(self.alpha) * 0.5
        up__bound = +sum(self.alpha) * 0.5
        step = 0.1
        threshold = numpy.arange(low_bound, up__bound, step)
        tprs = numpy.zeros(threshold.size, dtype = numpy.float16)
        fprs = numpy.zeros(threshold.size, dtype = numpy.float16)

        for t in range(threshold.size):
            output = self.prediction(self._mat, threshold[t])

            Num_tp = 0 # Number of true positive
            Num_fn = 0 # Number of false negative
            Num_tn = 0 # Number of true negative
            Num_fp = 0 # Number of false positive
            
            for i in range(self.samplesNum):
                if self._label[i] == LABEL_POSITIVE:
                    if output[i] == LABEL_POSITIVE:
                        Num_tp += 1
                    else:
                        Num_fn += 1
                else:
                    if output[i] == LABEL_POSITIVE:
                        Num_fp += 1
                    else:
                        Num_tn += 1

            tpr = Num_tp * 1.0/(Num_tp + Num_fn)
            fpr = Num_fp * 1.0/(Num_tn + Num_fp)

            tprs[t] = tpr
            fprs[t] = fpr

        fileObj = open(ROC_FILE, "a+")
        
        for t, f, th in zip(tprs, fprs, threshold):
            fileObj.write(str(t) + "\t" + str(f) + "\t" + str(th) + "\n")

        fileObj.flush()
        fileObj.close()

        pyplot.title("The ROC curve")
        pyplot.plot(fprs, tprs, "-r", linewidth = 1)
        pyplot.xlabel("fpr")
        pyplot.ylabel("tpr")
        pyplot.axis([-0.02, 1.1, 0, 1.1])
        pyplot.show()
        
        
    def saveModel(self, filename):
        fileObj = open(filename, "a+")

        for m in range(self.N):
            fileObj.write(str(self.alpha[m]) + "\n")
            fileObj.write(str(self.G[m].opt_dimension) + "\n")
            fileObj.write(str(self.G[m].opt_direction) + "\n")
            fileObj.write(str(self.G[m].opt_threshold) + "\n")

        fileObj.flush()
        fileObj.close()