"""
Created on Sun Mar 22 20:11:01 2020

@author: Poorvi Rai
"""

from Config import TEST_IMG
from Detector import Detector

from matplotlib import image
from time import time

start_time = time()

img = image.imread(TEST_IMG)

if len(img.shape) == 3:
    imgSingleChannel = img[:,:, 1]
else:
    imgSingleChannel = img

det = Detector()

rectangles = det.scanImgOverScale(imgSingleChannel)

end_time = time()

print("Number of rectangles: ", len(rectangles))
print("Cost time: ", end_time - start_time)

det.showResult(img, rectangles)
