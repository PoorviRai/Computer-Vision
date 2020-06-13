"""
Created on Thu Mar 19 21:19:53 2020

@author: Poorvi Rai
"""

from Config import HAAR_FEATURE_TYPE_I, HAAR_FEATURE_TYPE_II, HAAR_FEATURE_TYPE_III, HAAR_FEATURE_TYPE_IV, HAAR_FEATURE_TYPE_V
from PreProcess import IntegrateImage

import numpy


class Feature:
    def __init__(self, img_Width, img_Height):

        self.featureName = "Haar Feature"

        self.img_Width = img_Width
        self.img_Height = img_Height

        self.tot_pixels = img_Width * img_Height

        self.featureTypes = (HAAR_FEATURE_TYPE_I, HAAR_FEATURE_TYPE_II, HAAR_FEATURE_TYPE_III, HAAR_FEATURE_TYPE_IV, HAAR_FEATURE_TYPE_V)

        self.features = self._evalFeatures_total()
        self.featuresNum = len(self.features)

        self.vector = numpy.zeros(self.featuresNum, dtype = numpy.float32)

        self.idxVector_tmp_0 = numpy.zeros(self.tot_pixels, dtype = numpy.int8)
        self.idxVector_tmp_1 = numpy.zeros(self.tot_pixels, dtype = numpy.int8)
        self.idxVector_tmp_2 = numpy.zeros(self.tot_pixels, dtype = numpy.int8)
        self.idxVector_tmp_3 = numpy.zeros(self.tot_pixels, dtype = numpy.int8)


    def vecRectSum(self, idxVector, x, y, width, height):
        idxVector *= 0 
        
        if x == 0 and y == 0:
            idxVector[width * height + 2] = +1
        elif x == 0:
            idx1 = self.img_Height * (width - 1) + height + y - 1
            idx2 = self.img_Height * (width - 1) + y - 1
            
            idxVector[idx1] = +1
            idxVector[idx2] = -1
        elif y == 0:
            idx1 = self.img_Height * (x + width - 1) + height - 1
            idx2 = self.img_Height * (x - 1) + height - 1
            
            idxVector[idx1] = +1
            idxVector[idx2] = -1
        else:
            idx1 = self.img_Height * (x + width - 1) + height + y - 1
            idx2 = self.img_Height * (x + width - 1) + y - 1
            idx3 = self.img_Height * (x - 1) + height + y - 1
            idx4 = self.img_Height * (x - 1) + y - 1

            assert idx1 < self.tot_pixels and idx2 < self.tot_pixels 
            assert idx3 < self.tot_pixels and idx4 < self.tot_pixels 

            idxVector[idx1] = + 1
            idxVector[idx2] = - 1
            idxVector[idx3] = - 1
            idxVector[idx4] = + 1

        return idxVector


    def FeatureTypeI(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y + height, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vecImg) - vec2.dot(vecImg))/featureSize


    def FeatureTypeII(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y, width, height)

        featureSize = width * height * 2

        return (vec1.dot(vecImg) - vec2.dot(vecImg))/featureSize


    def FeatureTypeIII(self,vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x +   width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x + 2*width, y, width, height)

        featureSize = width * height * 3

        return (vec1.dot(vecImg) - vec2.dot(vecImg)-  vec3.dot(vecImg))/featureSize


    def FeatureTypeIV(self,vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x, y +   height, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x, y + 2*height, width, height)

        featureSize = width * height * 3

        return (vec1.dot(vecImg) - vec2.dot(vecImg) - vec3.dot(vecImg))/featureSize


    def FeatureTypeV(self, vecImg, x, y, width, height):
        vec1 = self.vecRectSum(self.idxVector_tmp_0, x + width, y, width, height)
        vec2 = self.vecRectSum(self.idxVector_tmp_1, x, y, width, height)
        vec3 = self.vecRectSum(self.idxVector_tmp_2, x, y + height, width, height)
        vec4 = self.vecRectSum(self.idxVector_tmp_3, x + width, y + height, width, height)

        featureSize = width * height * 4

        return (vec1.dot(vecImg) - vec2.dot(vecImg) + vec3.dot(vecImg) - vec4.dot(vecImg))/featureSize


    def _evalFeatures_total(self):
        win_Height = self.img_Height
        win_Width  = self.img_Width

        height_Limit = {HAAR_FEATURE_TYPE_I : win_Height/2 - 1,
                        HAAR_FEATURE_TYPE_II : win_Height - 1,
                        HAAR_FEATURE_TYPE_III : win_Height - 1,
                        HAAR_FEATURE_TYPE_IV : win_Height/3 - 1,
                        HAAR_FEATURE_TYPE_V : win_Height/2 - 1}

        width_Limit  = {HAAR_FEATURE_TYPE_I : win_Width - 1,
                        HAAR_FEATURE_TYPE_II : win_Width/2 - 1,
                        HAAR_FEATURE_TYPE_III : win_Width/3 - 1,
                        HAAR_FEATURE_TYPE_IV : win_Width - 1,
                        HAAR_FEATURE_TYPE_V : win_Width/2 - 1}

        features = []
        
        for types in self.featureTypes:
            for w in range(1, int(width_Limit[types])):
                for h in range(1, int(height_Limit[types])):
                    if w == 1 and h == 1:
                        continue

                    if types == HAAR_FEATURE_TYPE_I:
                        x_limit = win_Width - w
                        y_limit = win_Height - 2 * h
                        
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append((types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_II:
                        x_limit = win_Width - 2 * w
                        y_limit = win_Height - h
                        
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append((types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_III:
                        x_limit = win_Width - 3 * w
                        y_limit = win_Height - h
                        
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append((types, x, y, w, h))


                    elif types == HAAR_FEATURE_TYPE_IV:
                        x_limit = win_Width - w
                        y_limit = win_Height - 3 * h
                        
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append((types, x, y, w, h))

                    elif types == HAAR_FEATURE_TYPE_V:
                        x_limit = win_Width - 2 * w
                        y_limit = win_Height - 2 * h
                        
                        for x in range(1, x_limit):
                            for y in range(1, y_limit):
                                features.append((types, x, y, w, h))

        return features


    def calFeatureForImg(self, img):
        assert isinstance(img, IntegrateImage)
        assert img.img.shape[0] == self.img_Height
        assert img.img.shape[1] == self.img_Width

        for i in range(self.featuresNum):
            type, x, y, w, h = self.features[i]

            if   type == HAAR_FEATURE_TYPE_I:
                self.vector[i] = self.FeatureTypeI(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_II:
                self.vector[i] = self.FeatureTypeII(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_III:
                self.vector[i] = self.FeatureTypeIII(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_IV:
                self.vector[i] = self.FeatureTypeIV(img.vecImg, x, y, w, h)
            elif type == HAAR_FEATURE_TYPE_V:
                self.vector[i] = self.FeatureTypeV(img.vecImg, x, y, w, h)
            else:
                raise Exception("unknown feature type")

        return self.vector