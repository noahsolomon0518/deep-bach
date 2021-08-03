from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import pickle
import random
from mido import MidiFile, MidiTrack, Message


if __name__ == "__main__":
    from datagen_config import *
else:
    from .datagen_config import *

"""Encodes a batch of inputs. Returns list of 18 input array of <batchsize> samples: 6 from previous events, 6 for current events, 6 for future events"""
def encodeXSamples(partitions, dt, oe):
    encoded = oe.fit_transform(partitions)
    length = len(encoded[0])
    previous = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)<dt]).reshape(-1,dt) for colInd in range(length)]
    current = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)==dt]).reshape(-1,1) for colInd in range(length)]
    future = [np.array([x for rowInd, x in enumerate(encoded[:,colInd]) if rowInd%(2*dt+1)>dt]).reshape(-1,dt) for colInd in range(length)]
    return previous + current + future

def encodeYSamples(y, ohe):
    return ohe.fit_transform(y)

def transposeKey(partition, halfSteps):
    partition = np.array(partition)[:,:4]
    partition[partition!="_"] =np.array(np.array(partition[partition!="_"], dtype=int) + halfSteps, dtype="U3")
    return partition








class Datagen:
    def __init__(self, dataPath, dt, transpositionRange, inputRanges = None, outputRanges = None, testSize = 0.10):
        self.dt = dt
        self.data = pd.read_csv(dataPath)
        self.testSize = testSize
        self.transpositionRange = transpositionRange
        
        self.oe = OrdinalEncoder(categories=self.getInputRanges()) if inputRanges == None else OrdinalEncoder(categories=inputRanges)
        self.ohe = OneHotEncoder(categories=self.getOutputRanges(), sparse=False) if inputRanges == None else OneHotEncoder(categories=outputRanges, sparse=False)
        self.indices = self.getIndices()
        self.trainIndices, self.testIndices = self.getTrainTestIndices()
        
    """Partitions all indices into train and test indices"""
    def getTrainTestIndices(self):
        random.shuffle(self.indices)
        nTrainIndices = int(len(self.indices)*(1-self.testSize))
        trainIndices = self.indices[:nTrainIndices]
        testIndices = self.indices[nTrainIndices:]
        return trainIndices, testIndices

    """Get all possible values for each column for ordinal encoder"""
    def getInputRanges(self):
        return getPitchRanges(self.data, self.transpositionRange) + getBeatRange() + getFermataRange()
    
    """Get all possible values for each column for one hot encoder"""
    def getOutputRanges(self):
        return getNetPitchRange(self.data, self.transpositionRange)

    """Get every possible index that can be used as a sample"""
    def getIndices(self):
        grouped = self.data.groupby("pieceID")
        offset = 0
        indices = []
        for i, group in grouped:
            indices.extend([offset + i for i in range(self.dt, len(group)-(self.dt+1))])
            offset = offset + len(group) 
        return indices
    
    


    """Encode batch based on given indices"""
    def getBatches(self, indices):
        batchsize = len(indices)
        sampleSize = 2*self.dt+1
        partitions = np.zeros((batchsize*sampleSize, 6), dtype="U3")
        y = np.zeros((batchsize, 1), dtype="U3")
        for i,index in enumerate(indices):
            partitions[i*(sampleSize):(i+1)*(sampleSize), :] = self.data.iloc[index-self.dt:index + self.dt + 1,:6]
            if(self.transpositionRange not in [None, (0,0)]):
                partitions[i*(sampleSize):(i+1)*(sampleSize), :4] = transposeKey(partitions[i*(sampleSize):(i+1)*(sampleSize), :4], np.random.randint(self.transpositionRange[0], self.transpositionRange[1]+1))
            partitions[i*(sampleSize)+(self.dt+1):(i+1)*(sampleSize),:] = np.flip(partitions[i*(sampleSize)+(self.dt+1):(i+1)*(sampleSize),:], axis = 0)
            maskedPartN = random.randint(0,3)
            y[i][0] = partitions[i*(sampleSize)+self.dt, maskedPartN]
            partitions[i*(sampleSize)+self.dt, maskedPartN] = "x"
        batchX = encodeXSamples(partitions, self.dt, self.oe)
        batchY = encodeYSamples(y, self.ohe)

        return batchX, batchY


    """Train datagen"""
    def getTrainBatches(self, batchsize=100):
        while True:
            sampleIndices = np.random.choice(self.trainIndices, size = batchsize)
            yield self.getBatches(sampleIndices)
        
    
    """Test datagen"""
    def getTestBatches(self, batchsize=100):
        while True:
            sampleIndices = np.random.choice(self.testIndices, size = batchsize)
            yield self.getBatches(sampleIndices)

    """Load datagen from ConfigLoader object from datagen_config.py"""
    @staticmethod
    def loadFromConfig(configLoader):
        return Datagen(**configLoader.config)

    @staticmethod
    def loadFromPreset(presetName):
        configLoader = ConfigLoader(presetName)
        return Datagen(**configLoader.config)

    """Save datagen as pickled object"""
    def save(self, _path):
        f = open(_path, "wb")
        pickle.dump(self, f)
        f.close()

    """Load serialized datagen"""
    @staticmethod
    def load(_path):
        f = open(_path, "rb")
        loaded = pickle.load(f)
        f.close()
        return loaded

###########
# Testing #
###########

dtInterval = 100
def dfPieceToMidi(dfPiece):
    mf = MidiFile()
    tracks = [MidiTrack() for i in range(4)]
    mf.tracks = tracks
    for i, track in enumerate(tracks):
        for msgInd, msg in enumerate(dfPiece.iloc[:,i]):
            if(msg not in ["_","x"]):
                track.append(Message("note_on", note = int(msg), time = 0))
                dt = 0
                c = 1
                while(msgInd+c < len(dfPiece.iloc[:,i])-1 and dfPiece.iloc[msgInd+c,i]=="_"):
                    dt+=dtInterval
                    c+=1
                dt+=dtInterval
                track.append(Message("note_off", note = int(msg), time = dt))  
    return mf

def fromBatchToDf(batch, oe, dt = 16):
    x, y = batch[0], batch[1]
    previous = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,dt) for i, inp in enumerate(x[:6])]
    current = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,1) for i, inp in enumerate(x[6:12])]
    future = [np.array([oe.categories_[i][int(ele)] for ele in inp.reshape(-1)]).reshape(-1,dt) for i, inp in enumerate(x[12:18])]
    data = np.zeros((len(previous[0]) * (2*dt+1), 6), dtype="U3")
    sampleLength = 2*dt+1
    for i in range(len(previous[0])):
        for colInd in range(6):
            data[i*sampleLength:i*sampleLength+dt,colInd] = previous[colInd][i]
            data[i*sampleLength+dt,colInd] = current[colInd][i][0]
            data[i*sampleLength+dt+1:(i+1)*sampleLength,colInd] = future[colInd][i]
    return data
    
#Treats one piece as many samples, encodes it, then decodes it back to a midi.
def testXSampleEncoding():
    datagen = Datagen("data/df/tokenized_normal.csv", 16, (0,0))
    trainDatagen = datagen.getTrainBatches(batchsize = 500)
    x = next(trainDatagen)
    pieces = [piece for i, piece in datagen.data.groupby("pieceID")]
    encodedPiece = encodeXSamples(pieces[0].iloc[:33*5,:6], 16, datagen.oe)
    back = fromBatchToDf([encodedPiece,None], datagen.oe)
    dfPieceToMidi(DataFrame(back)).save("test3.mid")


def testBatchEncoding():
    datagen = Datagen("data/df/tokenized_normal.csv", 16, (0,0))
    trainDatagen = datagen.getTrainBatches(batchsize = 500)
    x = next(trainDatagen)
    back = fromBatchToDf(x, datagen.oe)
    dfPieceToMidi(DataFrame(back)).save("test4.mid")


def testTranspose():
    datagen = Datagen("data/df/tokenized_normal.csv", 16, (-10,10))
    trainDatagen = datagen.getTrainBatches(batchsize = 500)
    x = next(trainDatagen)
    pieces = [piece for i, piece in datagen.data.groupby("pieceID")]
    piece = pieces[0].iloc[:33*5,:6]
    piece.iloc[:,:4] = transposeKey(piece, 8)
    encodedPiece = encodeXSamples(piece, 16, datagen.oe)
    back = fromBatchToDf([encodedPiece,None], datagen.oe)
    dfPieceToMidi(DataFrame(back)).save("test5.mid")

    piece = pieces[0].iloc[:33*5,:6]
    piece.iloc[:,:4] = transposeKey(piece, -8)
    encodedPiece = encodeXSamples(piece, 16, datagen.oe)
    back = fromBatchToDf([encodedPiece,None], datagen.oe)
    dfPieceToMidi(DataFrame(back)).save("test6.mid")

if __name__ == "__main__":
    testTranspose()