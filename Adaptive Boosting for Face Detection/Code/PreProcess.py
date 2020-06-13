"""
Created on Fri Mar 20 09:37:43 2020

@author: Poorvi Rai
"""

import numpy
import os
import pylab

from matplotlib import pyplot, image


class IntegrateImage:

    def __init__(self, fileName = None, label = None, Mat = None):
        if fileName != None:
            self.imgName = fileName
            self.img = image.imread(fileName)

            if len(self.img.shape) == 3:
                self.img = self.img[:,:, 1]

        else:
            assert Mat != None
            self.img = Mat

        self.label = label
        self.vecImg = IntegrateImage._integrateImg(IntegrateImage._normalization(self.img)).transpose().flatten()


    @staticmethod
    def _integrateImg(image):
        assert image.__class__ == numpy.ndarray

        row, col = image.shape
        iImg = numpy.zeros((row, col))
        iImg = image.cumsum(axis=1).cumsum(axis=0)
        
        return iImg


    @staticmethod
    def _normalization(image):
        assert image.__class__ == numpy.ndarray

        row, col = image.shape
        stdImg = numpy.zeros((row, col))
        meanVal = image.mean()
        stdValue = image.std()

        if stdValue == 0:
            stdValue = 1

        stdImg = (image - meanVal)/stdValue

        return stdImg


    @staticmethod
    def show(image = None):
        if image == None:
            return
        
        pyplot.matshow(image)
        pylab.show()


class ImageSet:
    def __init__(self, imgDir = None, label = None, sampleNum = None):
        assert isinstance(imgDir, str)

        self.imgDir = imgDir
        self.fileList = os.listdir(imgDir)
        self.fileList.sort()

        if sampleNum == None:
            self.sampleNum = len(self.fileList)
        else:
            self.sampleNum = sampleNum

        self.curFileIdx = self.sampleNum
        self.label = label

        self.images = [None for _ in range(self.sampleNum)]

        processed = -10.
        for i in range(self.sampleNum):
            self.images[i] = IntegrateImage(imgDir + self.fileList[i], label)

            if i % (self.sampleNum / 10) == 0:
                processed += 10.
                print ("Loading ", processed, "%")

        print ("Loading  100 %\n")


    def readNextImg(self):
        img = IntegrateImage(self.imgDir + self.fileList[self.curFileIdx], self.label)
        self.curFileIdx += 1
        
        return img
