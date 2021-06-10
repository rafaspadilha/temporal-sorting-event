##################################################
# Padilha et al., "Temporally sorting images from real-world events",
# Pattern Recognition Letters, 2021
#
# Code for testing a single BvA classifier on the binary BvA setting
#
# usage:
#       python testing_singleBvAClassifier.py <modelFilePath> <setAorB> <cutOffClass>
# params:
#		modelFilePath - the path to the model to be evaluated
#		setAorB - either 'setA' or 'setB'
#       cutOffClass - the cutoff Class for binary Before vs After model
##################################################

############################
# IMPORTS
############################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation
# config.gpu_options.visible_device_list = "6"  #set which GPU to use
set_session(tf.Session(config=config))

import sys
import numpy as np

from keras.models import load_model
from dataLoaderBinary import DataLoader
from utils import *

############################
# DEFINES
############################
modelFilePath = sys.argv[1]
setAorB = sys.argv[2]
cutoffClass = sys.argv[3]

############################
# LOAD MODEL
############################
print "\n ---> Loading network architecture and model ....\n"
model = load_model(modelFilePath)



print "\n ---> Setup DataLoader ... \n"
tl = DataLoader(cutoffClass)


print "\n ---> Testing ... \n"
nBefore, nAfter = 0, 0
cBefore, cAfter = 0.0, 0.0

for batch, labelList in tl.loadValBatch(setAorB, batchSize = 1):
    predList = model.predict_on_batch(batch)
    y_pred = np.argmax(predList)
    y_true = np.argmax(labelList)

    if y_true == 0:
        nBefore += 1
        if y_pred == 0:
            cBefore += 1
    elif y_true == 1:
        nAfter += 1
        if y_pred == 1:
            cAfter += 1


    tBeforeRate = cBefore / nBefore
    tAfterRAte = cAfter / nAfter
    accNorm = (tBeforeRate+tAfterRAte) / 2.0


print "AccNorm = ", accNorm
print "tBeforeRate = ", tBeforeRate
print "tAfterRage = ", tAfterRate
