##################################################
# Padilha et al., "Temporally sorting images from real-world events",
# Pattern Recognition Letters, 2021
#
# Code for testing the Non-Hierarchical Pipeline
#
# usage:
#       python testing_allClassifiers_NonHierarchical.py <setAorB> <cutOffClass>
# params:
#	setAorB - either 'setA' or 'setB' for testing
#       cutOffClass - the cutoff Class for binary Before vs After model
##################################################

############################
# IMPORTS
############################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
#config.gpu_options.per_process_gpu_memory_fraction = 0.32  #uncomment to limit GPU allocation
config.gpu_options.visible_device_list = "0"  #set which GPU to use
set_session(tf.Session(config=config))

import os, sys
import numpy as np
from keras.models import load_model
from dataLoaderMultiClass import DataLoaderTest
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from utils import *



############################
# DEFINES
############################
testSet = sys.argv[1]

#### SetA used for VAL / SetB used for TEST
if testSet == "setB":
	modelPaths = ["./models/cutoffClass_1/setA/w.epoch-0693_val_loss-0.138357.hdf5",
			"./models/cutoffClass_2/setA/w.epoch-0473_val_loss-0.124818.hdf5",
			"./models/cutoffClass_3/setA/w.epoch-0084_val_loss-0.175408.hdf5"]

#### SetB used for VAL / SetA used for TEST
else:
	modelPaths = ["./models/cutoffClass_1/setB/w.epoch-0264_val_loss-0.163483.hdf5",
                        "./models/cutoffClass_2/setB/w.epoch-0462_val_loss-0.138809.hdf5",
                        "./models/cutoffClass_3/setB/w.epoch-0141_val_loss-0.175573.hdf5"]

# For each testing image, we create <nAugmentation> augmented images and average their results
nAugmentation = 10

# Instantiating the DataLoader
dl = DataLoaderTest(ignoreNotSure = True)


############################
# LOAD MODELS
############################
print "\n ---> Loading models ....\n"

coDict = {'acc_norm': acc_norm, 'tAfter_rate': tAfter_rate, 'tBefore_rate': tBefore_rate}

# Loading each model
modelList = [load_model(modelFilePath, custom_objects=coDict) for modelFilePath in modelPaths]
print "Models loaded"



############################
# DATA AUGMENTER
############################
dataAugmenter = ImageDataGenerator(rotation_range = 10,
                                   horizontal_flip=True, 
                                   vertical_flip=False,
                                   fill_mode='nearest')


############################
# TESTING
############################

confMatrix = np.zeros((dl.nClasses, dl.nClasses), dtype=np.int32)

for batch, label in dl.loadMultiClassTestBatch(testSet, batchSize = 1):
        #Augment images and include them into the batch
	if nAugmentation > 0:
		augmentedImages = np.array([dataAugmenter.random_transform(batch[0]) for _ in range(nAugmentation)])
        	batch = np.vstack((batch,augmentedImages))
	
	
	combinedScoreList = []

        # For each model, predict each image in the batch and average their scores
	scores = [np.mean(model.predict_on_batch(batch), axis=0, dtype=np.float64) for model in modelList]

        # We will use only the 'after' score 
	scores = [s[1] for s in scores]

        # Combine the scores of all models
	for i in range(len(scores) + 1):
		combinedScore = np.prod(scores[:i]) * np.prod([1.0 - score for score in scores[i:]])
		combinedScoreList.append(combinedScore)

        # The class is the point in time with the max combined score
	pointInTime = int(np.argmax(combinedScoreList))

        # Label was encoded as an one-hot array, so we get its index
	lb = np.argmax(label)

        # Update the confusion matrix
	confMatrix[lb][pointInTime] += 1 	
	
        # Print it just to keep track of how the evaluation is going
	print confMatrix, "\n\n"


############################
# PRINT METRICS and CONFUSION MATRIX
############################
acc = float(np.sum(np.diag(confMatrix))) / float(np.sum(confMatrix))
print "Acc = ", acc

offByOne = offByOneAcc(confMatrix, dl.nClasses)
print "OffByOne ACC = ", offByOne


computeAndPrintMAE(confMatrix, dl.nClasses)

print formatConfMatrix(confMatrix)

