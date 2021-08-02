from keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
from itertools import chain
import numpy as np
import pickle
import random
from datagen_config import *



def encodeXSample(dfPartition, dt, oe):
    dt = int(len(dfPartition.iloc[:,0])/2)

    previous = np.transpose(oe.fit_transform(dfPartition.iloc[:dt,:]))
    current = np.transpose(oe.fit_transform(dfPartition.iloc[dt,:].to_numpy().reshape(1,-1)))
    future = np.transpose(oe.fit_transform(dfPartition.iloc[dt+1::-1,:]))
    return [np.array(inp) for timeBlock in [previous, current, future] for inp in timeBlock]

def encodeYSample(y, ohe):
    return ohe.fit_transform(y)[0]


tokenizedNormal = pd.read_csv("data/df/tokenized_normal.csv")

oeRange = getPitchRanges(tokenizedNormal) + getBeatRange() + getFermataRange()




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
        batchX = [np.array([]) for i in range(18)]
        batchY = [np.array([])]
        for index in indices:
            dfPartition = self.data.iloc[index-self.dt:index + self.dt + 1,:6].copy()
            print(index-self.dt,index + self.dt + 1 )

            maskedPartN = random.randint(0,3)
            y = np.array([[dfPartition.iloc[self.dt, maskedPartN]]])
            dfPartition.iloc[self.dt, maskedPartN] = "x"
            xSample = encodeXSample(dfPartition, self.oe)
            ySample = encodeYSample(y, self.ohe)
            for i, xIn in enumerate(xSample):
                batchX[i] = np.concatenate([batchX[i],xIn], axis = 0)
            batchY[0] = np.concatenate([batchY[0],ySample], axis = 0)
        batchX = [xIn.reshape(len(indices),-1) for xIn in batchX]
        batchY = [batchY[0].reshape(len(indices),-1)]
        return batchX, batchY

        


    def getTrainBatches(self, batchsize=100):
        while True:
            sampleIndices = np.random.randint(0,len(self.trainIndices), size = batchsize)
            yield self.getBatches(sampleIndices)


    #def getTestBatches(self, batchsize=100):



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


    
datagen = DeepBachDatagen(tokenizedNormal, 16, (0,0))
trainDatagen = datagen.getTrainBatches(batchsize = 100)
for i in range(20):

    print(next(trainDatagen))