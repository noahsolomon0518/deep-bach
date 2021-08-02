from ast import Str
from keras.utils import Sequence
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
from itertools import chain
import numpy as np
import pickle
import random
from datagen_config import *



def encodeXSamples(partitions, dt, oe):
    encoded = oe.fit_transform(partitions)
    length = len(encoded[0])
    previous = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)<dt]).reshape(-1,dt) for colInd in range(length)]
    current = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)==dt]).reshape(-1,1) for colInd in range(length)]
    future = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)>dt]).reshape(-1,dt) for colInd in range(length)]
    return previous + current + future

def encodeYSamples(y, ohe):
    return ohe.fit_transform(y)[0]









class DeepBachDatagen:
    def __init__(self, dataPath, dt, transpositionRange, inputRanges = None, outputRanges = None, testSize = 0.10):
        self.dt = dt
        self.data = pd.read_csv(dataPath)
        self.testSize = testSize
        self.transpositionRange = transpositionRange
        
        self.oe = OrdinalEncoder(categories=self.getInputRanges()) if inputRanges == None else OrdinalEncoder(categories=inputRanges)
        self.ohe = OneHotEncoder(categories=self.getOutputRanges(), sparse=False) if inputRanges == None else OneHotEncoder(categories=outputRanges)
        self.indices = self.getIndices()
        self.trainIndices, self.testIndices = self.getTrainTestIndices()
        

    def getTrainTestIndices(self):
        random.shuffle(self.indices)
        nTrainIndices = int(len(self.indices)*(1-self.testSize))
        trainIndices = self.indices[:nTrainIndices]
        testIndices = self.indices[nTrainIndices:]
        return trainIndices, testIndices


    def getInputRanges(self):
        return getPitchRanges(self.data, self.transpositionRange) + getBeatRange() + getFermataRange()
    
    def getOutputRanges(self):
        return getNetPitchRange(self.data, self.transpositionRange)

    def getIndices(self):
        grouped = self.data.groupby("pieceID")
        offset = 0
        indices = []
        for i, group in grouped:
            indices.extend([offset + i for i in range(self.dt, len(group)-(self.dt+1))])
            offset = offset + len(group) 
        return indices

    def getBatches(self, indices):
        batchsize = len(indices)
        sampleSize = 2*self.dt+1
        partitions = np.zeros((batchsize*sampleSize, 6), dtype="U3")
        y = np.zeros((batchsize, 1), dtype="U3")
        for i,index in enumerate(indices):
            partitions[i*(sampleSize):(i+1)*(sampleSize), :] = self.data.iloc[index-self.dt:index + self.dt + 1,:6]
            maskedPartN = random.randint(0,3)
            y[i][0] = partitions[i*(sampleSize)+self.dt, maskedPartN]
            partitions[i*(sampleSize)+self.dt, maskedPartN] = "x"
        batchX = encodeXSamples(partitions, self.dt, self.oe)
        batchY = encodeYSamples(y, self.ohe)

        return batchX, batchY



    def getTrainBatches(self, batchsize=100):
        while True:
            sampleIndices = np.random.choice(self.trainIndices, size = batchsize)
            yield self.getBatches(sampleIndices)
    
    def getTestBatches(self, batchsize=100):
        while True:
            sampleIndices = np.random.choice(self.testIndices, size = batchsize)
            yield self.getBatches(sampleIndices)





    @staticmethod
    def loadFromConfig(data, config):
        return DeepBachDatagen(data, **config)

    def save(self, _path):
        f = open(_path, "wb")
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(_path):
        f = open(_path, "rb")
        loaded = pickle.load(f)
        f.close()
        return loaded



def fromBatchToDf(batch, oe, dt = 16):
    x, y = batch[0], batch[1]
    previous = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,dt) for i, inp in enumerate(x[:6])]
    current = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,1) for i, inp in enumerate(x[6:12])]
    future = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,dt) for i, inp in enumerate(x[12:18])]
    data = np.zeros((len(previous) * (2*dt+1), 6), dtype="U3")
    sampleLength = 2*dt+1
    for i in range(len(previous)):
        for colInd in range(6):
            data[i*sampleLength:i*sampleLength+dt,colInd] = previous[colInd][i]
            data[i*sampleLength+dt,colInd] = current[colInd][i][0]
            data[i*sampleLength+dt+1:(i+1)*sampleLength,colInd] = future[colInd][i]
    
            

if __name__ == "__main__":
    datagen = DeepBachDatagen("data/df/tokenized_normal.csv", 16, (0,0))
    trainDatagen = datagen.getTrainBatches(batchsize = 500)
    x = next(trainDatagen)
    for i in range(10):
        next(trainDatagen)
        print(i)
