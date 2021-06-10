##################################################
# Padilha et al., "Temporally sorting images from real-world events", 
# Pattern Recognition Letters, 2021
#
# Training code for Non-Hierarchical Before vs After models
#
# usage: 
#       python training.py <cutOffClass>
# params:
#       cutOffClass - the cutoff Class for binary Before vs After model
##################################################



#### IMPORTS and DEFINES 

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True    #avoid getting all available memory in GPU
#config.gpu_options.per_process_gpu_memory_fraction = 0.5  #uncomment to limit GPU allocation
# config.gpu_options.visible_device_list = "6"  #set which GPU to use
set_session(tf.Session(config=config))


import numpy as np
import os, sys

from utils import *

from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, TensorBoard

from dataLoaderBinary import DataLoader


#### Input Param
cutoffClass = sys.argv[1]



#### Other definitions
modelDirectory = os.path.join('./models', "cutoffClass_" + cutoffClass)  #path to save the checkpoints
batchSize = 30




############################
# SETUP
############################
dl = DataLoader(int(cutoffClass))

# Create path to save models and logs
logPath = os.path.join(modelDirectory, "logs")
if not os.path.exists(logPath):
        os.makedirs(logPath)


############################
# LOAD MODEL
############################
print "\n ---> Loading network architecture and model ....\n"

baseModel = InceptionResNetV2(include_top = False, weights="imagenet", input_shape = (299,299,3), pooling = 'avg')

for layer in baseModel.layers:
	if layer.name == "conv_7b":
		break
	layer.trainable = False

output = Dropout(0.7)(baseModel.output)
output = Dense(2, activation = 'softmax')(output)
model = Model(baseModel.input, output)






############################
# TRAINING
############################

#Define training optimization parameters
model.compile(loss="categorical_crossentropy", 
                optimizer=Adam(lr=0.0001), 
                metrics=[acc_norm, tBefore_rate, tAfter_rate]) #metrics defined in utils.py

checkpointer = ModelCheckpoint(filepath= os.path.join(modelDirectory, "w.epoch-{epoch:04d}.hdf5"), verbose=0)
tensorboarder =  TensorBoard(log_dir=logPath)

model.fit_generator(dl.loadTrainBatch("train", batchSize = batchSize),
                    steps_per_epoch = (900 / batchSize),  #900 is roughly the number of training images
                    epochs= 1000, verbose=1,
                    callbacks = [checkpointer, tensorboarder])



