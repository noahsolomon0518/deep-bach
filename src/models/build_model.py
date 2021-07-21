
from tensorflow.python.keras.layers.core import Dropout
from create_datagen import Datagen
import pandas as pd
from itertools import chain
from tqdm import tqdm
from keras.models import Model, Input, load_model
from keras.layers import Dense, Embedding, concatenate, LSTM, Flatten
import numpy as np
import pickle 


def buildModel(
    inputDimensions, embeddingMaxDimensions, embeddingOutputDimension, LSTM1Dimension, 
    middleDimension, softmaxDimension, LSTM1Dropout = 0, LSTM1RecurrentDropout = 0, middleDropout = None,
    LSTM2Dimension = None, LSTM2Dropout = 0, LSTM2RecurrentDropout = 0,
    ):
    
    modelInputs = [Input(shape = (inpDim,)) for inpDim in inputDimensions]
    embeddings= [Embedding(dim, embeddingOutputDimension) for dim in embeddingMaxDimensions]

    leftLSTMIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(5)])
    rightLSTMIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(10,15)])
    if(LSTM2Dimension!=None):
        leftLSTMOut = LSTM(LSTM1Dimension, return_sequences = True, recurrent_dropout=LSTM1RecurrentDropout, dropout = LSTM1Dropout)(leftLSTMIn)
        leftLSTMOut = Flatten()(LSTM(LSTM2Dimension, recurrent_dropout=LSTM2RecurrentDropout, dropout = LSTM2Dropout)(leftLSTMOut))

        rightLSTMOut = LSTM(LSTM1Dimension, return_sequences = True, recurrent_dropout=LSTM1RecurrentDropout, dropout = LSTM1Dropout)(rightLSTMIn)
        rightLSTMOut = Flatten()(LSTM(LSTM2Dimension, recurrent_dropout=LSTM2RecurrentDropout, dropout = LSTM2Dropout)(rightLSTMOut))

    else:
        leftLSTMOut = Flatten()(LSTM(LSTM1Dimension, recurrent_dropout=LSTM1RecurrentDropout, dropout = LSTM1Dropout)(leftLSTMIn))
        rightLSTMOut = Flatten()(LSTM(LSTM1Dimension, recurrent_dropout=LSTM1RecurrentDropout, dropout = LSTM1Dropout)(rightLSTMIn))

    middleIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(5,10)])
    middleOut = Flatten()(Dense(middleDimension, activation = 'relu')(middleIn))
    if(middleDropout!=None):
        middleOut = Dropout(middleDropout)(middleOut)
    

    beforeSoftmax = concatenate([leftLSTMOut, middleOut, rightLSTMOut])
    modelOutput = Dense(softmaxDimension, activation = 'softmax')(beforeSoftmax)
    model = Model(modelInputs, modelOutput)
    model.summary()
    return Model(modelInputs, modelOutput)



def model1():
    trainDf = pd.read_csv("../../data/train_df.csv")
    testDf = pd.read_csv("../../data/test_df.csv")
    oeranges = [trainDf.append(testDf)[col].unique().tolist() for col in trainDf.columns[:5]]
    for i in range(4):
        oeranges[i].append("x")
    oheranges = [sorted(list(set(chain.from_iterable([df[column].unique().tolist() for df in [trainDf, testDf] for column in df.columns[:4]]))))]
    datagenTrain = Datagen(trainDf, 8, 5, 32, oeranges, oheranges)
    datagenTest = Datagen(testDf, 8, 5, 32, oeranges, oheranges)
    testX,testY = datagenTest[0]
    
    maxdim = [len(oerange) for oerange in oeranges]
    embedDim = 16

    modelInputs = [Input(shape = (xInp.shape[1],)) for xInp in testX]
    embeddings= [Embedding(dim, embedDim) for dim in maxdim]

    leftLSTMIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(5)])
    leftLSTMOut = Flatten()(LSTM(128)(leftLSTMIn))

    middleIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(5,10)])
    middleOut = Flatten()(Dense(64, activation = 'relu')(middleIn))

    rightLSTMIn = concatenate([embeddings[i%5](modelInputs[i]) for i in range(10,15)])
    rightLSTMOut = Flatten()(LSTM(128)(rightLSTMIn))

    beforeSoftmax = concatenate([leftLSTMOut, middleOut, rightLSTMOut])
    modelOutput = Dense(47, activation = 'softmax')(beforeSoftmax)

    model = Model(modelInputs, modelOutput)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])
    model.fit(datagenTrain, validation_data = datagenTest, validation_steps = 20, steps_per_epoch = 90, epochs = 100)
    model.save("../../models/model1/model1.h5")
    xysamplerPath = open("../../models/model1/xysampler.pickle", "wb")
    pickle.dump(datagenTest._XYSampler, xysamplerPath)
    xysamplerPath.close()


    





if __name__ == "__main__":
    #model1()
    #predict("../../models/model1.h5")
    #testModel()
    pass






