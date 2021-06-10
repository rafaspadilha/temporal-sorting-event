from prettytable import PrettyTable
import keras.backend as K
import numpy as np

############################
# METRICS
############################
def tAfter_rate(y_true, y_pred):
        total_after = K.sum(K.tf.cast(K.tf.equal(y_true[:, 1], 1), 'float32'))
        tAfter = K.tf.equal(y_true[:, 1], 1) & K.tf.equal(
            K.argmax(y_pred, axis=-1), 1)
        tAfter = K.sum(K.tf.cast(K.reshape(tAfter, [-1]), 'float32'))

        if total_after == 0:
                return 0.0
        else:
                return tAfter/total_after


def tBefore_rate(y_true, y_pred):
        total_before = K.sum(K.tf.cast(K.tf.equal(y_true[:, 0], 1), 'float32'))
        tBefore = K.tf.equal(y_true[:, 0], 1) & K.tf.equal(
            K.argmax(y_pred, axis=-1), 0)
        tBefore = K.sum(K.tf.cast(K.reshape(tBefore, [-1]), 'float32'))

        if total_before == 0:
                return 0.0
        else:
                return tBefore/total_before


def acc_norm(y_true, y_pred):
        TAfterRate = tAfter_rate(y_true, y_pred)
        TBeforeRate = tBefore_rate(y_true, y_pred)
        return (TAfterRate + TBeforeRate)/2.0


############################
# Pretty Table aux method
############################

def formatConfMatrix(confMatrix):
        classes = ["Sp on Fire", "Sp Collapsing", "No Sp", "Fire Ext"]
        table = PrettyTable([""] + classes, junction_char='|')

        for rowIdx in range(len(classes)):
                row = [classes[rowIdx]] + [confMatrix[rowIdx][colIdx]
                                           for colIdx in range(len(classes))]
                table.add_row(row)

        return table


############################
# Method to calculate the off-by-one accuracy
############################
def offByOneAcc(confMatrix, nClasses):
        correctSamples = 0.0
        for idx in range(nClasses):
                correctSamples += confMatrix[idx][idx]
                if idx > 0:
			correctSamples += confMatrix[idx][(idx - 1) % (nClasses)]
		if idx < nClasses - 1:
               		correctSamples += confMatrix[idx][(idx + 1) % (nClasses)]

        return correctSamples / float(np.sum(confMatrix))


############################
# Method to calculate and print the MAE and balanced MAE
############################
def dMatrix(i, j):
        return abs(i-j)


def getMAE(confMatrix, nClasses):
        absError = confMatrix * np.fromfunction(dMatrix, (nClasses, nClasses))
        mae = np.sum(absError) / np.sum(confMatrix)
        return mae


def getMAEBalanced(confMatrix, nClasses):
        absError = confMatrix * np.fromfunction(dMatrix, (nClasses, nClasses))
        perClassMAE = np.sum(absError, axis=1) / np.sum(confMatrix, axis=1)
        return np.mean(perClassMAE)


def computeAndPrintMAE(confMatrix, nClasses):
        print "MAE = ", getMAE(confMatrix, nClasses)
        print "MAE Balanced = ", getMAEBalanced(confMatrix, nClasses)
