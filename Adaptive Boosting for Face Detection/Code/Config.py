"""
Created on Sun Mar 15 14:32:25 2020

@author: Poorvi Rai
"""

# training set directory for face and non-face images
TRAINING_FACE    = "E:\\Sem 4\\CV\\Projects\\Project02\\TrainingImages\\FACES\\"
TRAINING_NONFACE = "E:\\Sem 4\\CV\\Projects\\Project02\\TrainingImages\\NFACES\\"

# test set directory for face and non-face images
TEST_FACE        = "./TrainingImages/FACES/"
TEST_NONFACE     = "./TrainingImages/NFACES/"

TEST_IMG         = "./Test/mona_lisa_cut.png"

ADABOOST_CACHE_FILE = "./model/adaboost_classifier.cache0"
ROC_FILE            = "./model/roc.cache"

FIGURES             = "./figure/"

# image size in the training set 19 * 19
TRAINING_IMG_HEIGHT = 19
TRAINING_IMG_WIDTH  = 19

# How many different types of  Haar-feature
FEATURE_TYPE_NUM    = 5

# How many number of features that a single training image have
FEATURE_NUM = 37862

# number of positive and negative sample will be used in the training process
POSITIVE_SAMPLE     = 4800
NEGATIVE_SAMPLE     = 9000

SAMPLE_NUM = POSITIVE_SAMPLE + NEGATIVE_SAMPLE

TESTING_POSITIVE_SAMPLE = 20
TESTING_NEGATIVE_SAMPLE = 20

TESTING_SAMPLE_NUM = TESTING_NEGATIVE_SAMPLE + TESTING_POSITIVE_SAMPLE

LABEL_POSITIVE = +1
LABEL_NEGATIVE = -1

WHITE = 255
BLACK = 0

EXPECTED_TPR = 0.999
EXPECTED_FPR = 0.0005


# the threshold range of adaboost. (from -inf to +inf)
AB_TH_MIN   = -15
AB_TH_MAX   = +15

HAAR_FEATURE_TYPE_I     = "I"
HAAR_FEATURE_TYPE_II    = "II"
HAAR_FEATURE_TYPE_III   = "III"
HAAR_FEATURE_TYPE_IV    = "IV"
HAAR_FEATURE_TYPE_V     = "V"

AB_TH       = -3.0
SEARCH_WIN_STEP = 4

ADABOOST_LIMIT = 150

DETECT_START = 1.
DETECT_END   = 2.
DETECT_STEP  = 0.2
