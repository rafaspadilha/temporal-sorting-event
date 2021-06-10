##################################################
# Padilha et al., "Temporally sorting images from real-world events",
# Pattern Recognition Letters, 2021
#
# Data loader code for evaluating Non-Hierarchical models
# This data loader load images in the multiclass scenario for the final evaluation
#
##################################################

import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from scipy.misc import imresize
from keras.utils import to_categorical


#### This is a data loader responsible for loading/pre-processing images and
# creating the batches that will be fed to the network during testing
#
# Make sure the splitDir path is correct and points to the 'splits' folder of the dataset

class DataLoaderTest(object):
        splitDir = "../dataset/splits/"
 
        # Some images are labeled as "not sure" w.r.t. the moment/sub-episode they were captured
        # The param "ignoreNotSure" is used to filter them
        def __init__(self, ignoreNotSure = False, resize_shape = (340,340,3), crop_size = (299,299)):
                self.resize_shape = resize_shape
                self.crop_size = crop_size
                self.ignoreNotSure = ignoreNotSure
                self._setDatasetPaths()


        def _setDatasetPaths(self):
                self.trainDataFile = os.path.join(self.splitDir,"train_temporal.txt")
                self.setADataFile = os.path.join(self.splitDir,"setA_temporal.txt")
                self.setBDataFile = os.path.join(self.splitDir, "setB_temporal.txt")
                if self.ignoreNotSure == True:
                        self.nClasses = 4
                else:
                        self.nClasses = 5

        def _readDataFile(self, set):
                data = []
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
                        data.append((imgPath, label))

                f.close()

                return data

        #Reads the split file and returns a list of perClass tuples of (path, classIdx) 
        def _readAndSplitDataInClasses(self, set):
                data = self._readDataFile(set)
                perClassData = [ [dataSample for dataSample in data if dataSample[1] == classIdx] for classIdx in range(self.nClasses)]
                return perClassData

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

                if mode == "train":  # random crop if training
                        x = np.random.randint(0, width - dx + 1)
                        y = np.random.randint(0, height - dy + 1)

                elif mode == "test" or mode == "val":  # center crop if val/test
                        x = (width/2) - (dx/2)
                        y = (height/2) - (dy/2)

                return img[y:(y+dy), x:(x+dx), :]


        # Main method that generates testing batches
        # Inputs:
        ### setToLoad - one of ['train', 'val', 'test']
        ### batchSize - number of images to load for each batch
        def loadMultiClassTestBatch(self, setToLoad, batchSize=10):
                perClassData = self._readAndSplitDataInClasses(setToLoad)

                batch, labelList = [],[]
                nInBatch = 0

                for classIdx in range(self.nClasses):
                        classData = perClassData[classIdx]
                        for sampleIdx in range(len(classData)):
                                imgPath, label = classData[sampleIdx]

                                img = self._loadImage(imgPath)
                                img = self._resizeAndCrop(img, "val")

                                batch.append(img)
                                labelList.append(to_categorical(label, num_classes=self.nClasses))

                                nInBatch += 1
                                if nInBatch >= batchSize:
                                                yield (np.array(batch), np.array(labelList))
                                                batch, labelList = [],[]
                                                nInBatch = 0
                
                if nInBatch >= batchSize:
                        yield (np.array(batch), np.array(labelList))


# Just a '__main__' to test batch generation
if __name__ == '__main__':
        l = DataLoader(task="temporal") 
        print l.nClasses

        for batch, labels in l.loadMultiClassTrainBatch("setA", 10):
                print batch.shape, labels.shape

                print labels
                exit()
