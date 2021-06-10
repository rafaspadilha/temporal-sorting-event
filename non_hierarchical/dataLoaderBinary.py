##################################################
# Padilha et al., "Temporally sorting images from real-world events",
# Pattern Recognition Letters, 2021
#
# Data loader code for Non-Hierarchical Before vs After models
# This data loder loads images in the binary scenario
#
##################################################

import os
import numpy as np
from random import shuffle

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from scipy.misc import imresize
from keras.utils import to_categorical


#### This is a data loader responsible for loading/pre-processing images and 
# creating the batches that will be fed to the network during training/testing
#
# Make sure the splitDir path is correct and points to the 'splits' folder of the dataset

class DataLoader(object):
        splitDir = "../dataset/splits/"
 
        def __init__(self, cutoffClass, ignoreClasses = [], resize_shape = (340,340,3), crop_size = (299,299)):
                self._setDatasetPaths()

                # Set up an ImageDataGenerator with augmentation (only applied in training)
                self.dataAugmenter = ImageDataGenerator(rotation_range = 10,
                                        horizontal_flip=True, vertical_flip=False,
                                        fill_mode='nearest')
                self.resize_shape = resize_shape
                self.crop_size = crop_size

                # As we train Binary Before vs After models, the cutoffClass determines
                # which classes will be assigned to the 'Before' or 'After' class
                self.cutoffClass = cutoffClass

                # ignoreClasses is useful for training the hierarchical method 
                self.ignoreClasses = ignoreClasses
                
                self.nClasses = 2

        def _setDatasetPaths(self):
                self.trainDataFile = os.path.join(self.splitDir,"train_temporal.txt")
                self.setADataFile = os.path.join(self.splitDir,"setA_temporal.txt")
                self.setBDataFile = os.path.join(self.splitDir, "setB_temporal.txt")



        def _readDataFile(self, set):
                beforeData, afterData = [], []
                if set == "train":
                        f = open(self.trainDataFile, "r")
                if set == "setA":
                        f = open(self.setADataFile, "r")
                elif set == "setB":
                        f = open(self.setBDataFile, "r")

                lines = f.readlines()
                for line in lines:
                        content = line.strip().split("\t")

                        imgPath = content[0]
                        label = int(content[-1])

                        if label not in self.ignoreClasses:
                                if label < self.cutoffClass:
                                        beforeData.append(imgPath)
                                elif label >= self.cutoffClass:
                                        afterData.append(imgPath)
                f.close()
                return (beforeData, afterData)


        # Load the image and use the network preprocessing function
        def _loadImage(self, imgPath):
                img = img_to_array(load_img(imgPath))
                img = preprocess_input(img) 
                return img 



        def _resizeAndCrop(self, img, mode = "train"):
                height, width = img.shape[0], img.shape[1]

                #Must resize if one dim is smaller then target_dim
                #but if both are bigger, we randomly choose if we will resize or not
                if width < self.resize_shape[1] or height < self.resize_shape[0] or (mode == "test" or mode == "val"):
                        factor = 1.0

                #During training, we perform a random scale and crop
                elif mode == "train":
                        factor = 1.0 + (np.random.random_sample()/2.0) #factor will be from 1.0 to 1.5

                if width <= height:
                        new_width = int(self.resize_shape[1] * factor)
                        new_height = (height * new_width) / (width)
                else:
                        new_height = int(self.resize_shape[0] * factor)
                        new_width = (width * new_height) / (height)

                img = imresize(img, (new_height, new_width, self.resize_shape[2]))



                # Cropping
                height, width = img.shape[0], img.shape[1]
                dy, dx = self.crop_size

                if mode == "train": #random crop if training
                        x = np.random.randint(0, width - dx + 1)
                        y = np.random.randint(0, height - dy + 1)

                elif mode == "test" or mode == "val": #center crop if val/test
                        x = (width/2) - (dx/2)
                        y = (height/2) - (dy/2)

                return img[y:(y+dy), x:(x+dx), :]


        # Main method that generates training batches
        # batches are balanced with half 'before' and half 'after' images
        # Inputs:
        ### setToLoad - one of ['train', 'val', 'test']
        ### batchSize - number of images to load for each batch
        def loadTrainBatch(self, setToLoad, batchSize = 10):
                beforeData, afterData = self._readDataFile(setToLoad)

                print "Before Data: ", len(beforeData)
                print "After Data: ", len(afterData)

                shuffle(beforeData)
                shuffle(afterData)

                batch, labelList = [],[]
                beforeProgress, afterProgress = 0, 0
                nInBatch = 0

                while 1:
                        imgPath = beforeData[beforeProgress]

                        img = self._loadImage(imgPath)
                        img = self._resizeAndCrop(img, "train")
                        img = self.dataAugmenter.random_transform(img)

                        batch.append(img)
                        labelList.append(to_categorical(0, num_classes=2))

                        beforeProgress += 1

                        if beforeProgress >= len(beforeData):
                                beforeProgress = 0
                                shuffle(beforeData)



                        imgPath = afterData[afterProgress]

                        img = self._loadImage(imgPath)
                        img = self._resizeAndCrop(img, "train")
                        img = self.dataAugmenter.random_transform(img)

                        batch.append(img)
                        labelList.append(to_categorical(1, num_classes=2))

                        afterProgress += 1

                        if afterProgress >= len(afterData):
                                afterProgress = 0
                                shuffle(afterData)		

        
                        nInBatch += 2

                        if nInBatch >= batchSize:
                                        yield (np.array(batch), np.array(labelList))
                                        batch, labelList = [],[]
                                        nInBatch = 0



        # Method to generate batches with all images from Val or Test sets
        # images are selected in order ('before', then 'after') with no
        # augmentation applied    
        def loadValBatch(self, setToLoad, batchSize = 10):
                beforeData, afterData = self._readDataFile(setToLoad)

                batch, labelList = [],[]
                nInBatch = 0

                for idx in range(2):
                        data = [beforeData, afterData][idx]

                        for sampleIdx in range(len(data)):
                                imgPath = data[sampleIdx]

                                img = self._loadImage(imgPath)
                                img = self._resizeAndCrop(img, "val")

                                batch.append(img)
                                labelList.append(to_categorical(idx, num_classes=2))

                                nInBatch += 1
                                if nInBatch >= batchSize:
                                                yield (np.array(batch), np.array(labelList))
                                                batch, labelList = [],[]
                                                nInBatch = 0
        
                if nInBatch >= batchSize:
                        yield (np.array(batch), np.array(labelList))





# Just a '__main__' to test batch generation
if __name__ == '__main__':
        l = DataLoader(cutoffClass = 1)

        for batch, labels in l.loadTrainBatch("setA", 10):
                print batch.shape, labels.shape

                print labels
                exit()
