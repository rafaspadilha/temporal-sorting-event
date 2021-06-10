##################################################
# Padilha et al., "Temporally sorting images from real-world events",
# Pattern Recognition Letters, 2021
#
# Code for testing the Hierarchical Pipeline
#
# usage:
#       python testing_allClassifiers_Hierarchical.py <setAorB>
# params:
#	setAorB - either 'setA' or 'setB' for testing
#
#
# For the top-level classifier in the Hierarchical Pipeline
# we use the same weights from the Non-Hierarchical model with
# cutoff class = 2 
##################################################

############################
# IMPORTS
############################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
# config.gpu_options.per_process_gpu_memory_fraction = 0.32  #uncomment to limit GPU allocation
# config.gpu_options.visible_device_list = "2"  #set which GPU to use
set_session(tf.Session(config=config))

import sys
import numpy as np
from keras.models import load_model
from dataLoaderTest import DataLoaderTest
from keras.preprocessing.image import ImageDataGenerator
from utils import *





############################
# DEFINES
############################
testSet = sys.argv[1]

#### SetA used for VAL / SetB used for TEST
if testSet == "setB":
        topLevelPath = "../non_hierarchical/models/cutoffClass_2/setA/w.epoch-0473_val_loss-0.124818.hdf5"
        beforeSpirePath = "./models/cutoffClass_1/setA/w.epoch-0088_val_loss-0.232605.hdf5"
        afterSpirePath = "./models/cutoffClass_3/setA/w.epoch-0035_val_loss-0.086825.hdf5"

#### SetB used for VAL / SetA used for TEST
else:
        topLevelPath = "../non_hierarchical/models/cutoffClass_2/setB/w.epoch-0462_val_loss-0.138809.hdf5"
        beforeSpirePath = "./models/cutoffClass_1/setB/w.epoch-0185_val_loss-0.240775.hdf5"
        afterSpirePath = "./models/cutoffClass_3/setB/w.epoch-0349_val_loss-0.061852.hdf5"

# For each testing image, we create <nAugmentation> augmented images and average their results
nAugmentation = 10

# Instantiating the DataLoader
dl = DataLoaderTest(ignoreNotSure=True)




############################
# LOAD MODELS
############################
print "\n ---> Loading models ....\n"

coDict = {'acc_norm': acc_norm, 'tAfter_rate': tAfter_rate, 'tBefore_rate': tBefore_rate}

# Loading each model
topLevelModel = load_model(topLevelPath, custom_objects=coDict)
beforeSpireModel = load_model(beforeSpirePath, custom_objects=coDict)
afterSpireModel = load_model(afterSpirePath, custom_objects=coDict)
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
        augmentedImages = np.array([dataAugmenter.random_transform(batch[0]) for _ in range(nAugmentation)])
        batch = np.vstack((batch,augmentedImages))
        

        # Predict image with topLevel model to decide which group of models will classify it
        topLevelPred = np.mean(topLevelModel.predict_on_batch(batch), axis=0, dtype=np.float64)

        if topLevelPred[1] < 0.5: #Before Spire Collapsed
                beforePred = np.mean(beforeSpireModel.predict_on_batch(batch), axis=0, dtype=np.float64)
                pointInTime = int(np.argmax(beforePred))

        else: #After Spire Collapsed
                afterPred = np.mean(afterSpireModel.predict_on_batch(batch), axis=0, dtype=np.float64)
                pointInTime = 2 + int(np.argmax(afterPred))

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

