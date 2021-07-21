
from numpy.lib.arraysetops import unique
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from random import randint
from keras.utils import Sequence
import numpy as np
from itertools import chain
import pandas as pd
from tqdm import tqdm
import argparse
from itertools import chain
import pickle

my_parser = argparse.ArgumentParser()






def uniquePerColumnInDf(df):
    return [list(df[column].unique()) for column in df.columns]



def uniquePerColumnInDfWithExtra(df, extra, columns):
    uniques = uniquePerColumnInDf(df)
    return [list(sorted(uni + extra)) for i,uni in enumerate(uniques) if i in columns]


def allUniqueInColumns(df):
    return [list(sorted(chain.from_iterable([df[column].unique() for column in df.columns])))]
    



class XYSampler:
    
    def __init__(self, df, ordinalEncoder, oneHotEncoder, dt):
        self.df = df.copy()
        self.ohe = oneHotEncoder
        self.oe = ordinalEncoder
        self.dt = dt
        
        
        #print(encoder.inverse_transform(np.concatenate([past, current.reshape(1,-1), np.array(list(reversed(future)))], axis = 0)))
        #print(encoder.inverse_transform(encoded) ==  encoder.inverse_transform(np.concatenate([past, current.reshape(1,-1), np.array(list(reversed(future)))], axis = 0)))

    def getSample(self, index):
        data = self.df.iloc[index-self.dt:index+self.dt+1,:-2].copy()
        masker = randint(0,3)
        index = self.dt
        y = self.ohe.fit_transform(np.array(data.iloc[index, masker]).reshape(1,1))
        data.iloc[index, masker] = "x"
        past, current, future = self.encodeX(data, self.dt)
        x = \
        [x for x in np.transpose(past)] + \
        [[x] for x in np.transpose(current)] + \
        [x for x in np.transpose(future)]
        return x,y
    
    def encodeX(self, dfX, dt):
        encoded =  self.oe.fit_transform(dfX)
        past = encoded[:dt,:]
        current = encoded[dt,:]
        future = np.array(list(reversed(encoded[dt+1:2*dt+1])))
        return past, current, future
        



class Datagen(Sequence):
    
    def __init__(self, df, dt, gap, batchsize, sampler):
        self.batchsize = batchsize
        self.dt = dt
        self.gap = gap
        self.indices = self.getIndices(df)
        self.sampler = sampler
    
    def getIndices(self, df):
        grouped = df.groupby("pieceID")
        offset = 0
        indices = []
        for i, group in grouped:
            indices.extend([offset + i for i in range(self.dt, len(group)-(self.dt), self.gap)])
            offset = offset + len(group) 
        return indices
    
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        batchIndices = np.random.choice(self.indices, self.batchsize, replace = False)
        x = [[] for i in range(15)]
        y = []
        for ind in batchIndices:
            curX, curY = self.sampler.getSample(ind)
            for i in range(15):
                x[i].append(curX[i])
            y.append(curY[0])
            
        for i in range(15):
            x[i] = np.stack(x[i])
        y = np.stack(y)
        return x,y
    
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



        

DATAGEN_PATH = "data/datagen/"
DF_PATH = "data/df/"
if (__name__ == "__main__"):
    my_parser.add_argument("train data", help = "CSV path of train data relative to data/df")
    my_parser.add_argument("test data", help = "CSV path of test data relative to data/df")
    my_parser.add_argument("train save path", help = "Where the train datagen will be saved relative to data/datagen")
    my_parser.add_argument("test save path", help = "Where the test datagen will be saved relative to data/datagen")
    my_parser.add_argument("dt", help = "Amount of dt datagen will create for samples")
    my_parser.add_argument("gap", help = "Gap between each potential sample")
    args = my_parser.parse_args()
    args = vars(args)

    trainDfPath = DF_PATH + args["train data"]
    testDfPath = DF_PATH + args["test data"]
    trainDatagenPath = DATAGEN_PATH + args["train save path"]
    testDatagenPath = DATAGEN_PATH + args["test save path"]

    dt = int(args["dt"])
    gap = int(args["gap"])

    trainDf = pd.read_csv(trainDfPath)
    testDf = pd.read_csv(testDfPath)



    combinedDf = trainDf.append(testDf)



    ordinalEncoderCategories = uniquePerColumnInDfWithExtra(combinedDf, ["x"], [0,1,2,3]) + [combinedDf.iloc[:,4].unique()]
    oneHotEncoderCategories = allUniqueInColumns(combinedDf.iloc[:,:4])

    trainSampler = XYSampler(trainDf, OrdinalEncoder(categories=ordinalEncoderCategories), OneHotEncoder(categories=oneHotEncoderCategories, sparse=False), int(dt))
    testSampler = XYSampler(testDf, OrdinalEncoder(categories=ordinalEncoderCategories), OneHotEncoder(categories=oneHotEncoderCategories, sparse=False), int(dt))

    trainDatagen = Datagen(trainDf, dt, gap, sampler = trainSampler, batchsize = 32)
    testDatagen = Datagen(testDf, dt, gap, sampler = testSampler, batchsize = 32)

    trainDatagen.save(trainDatagenPath)
    testDatagen.save(testDatagenPath)

    print("Train and test datagen's successfully saved.")



