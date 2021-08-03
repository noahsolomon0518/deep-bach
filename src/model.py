from keras.models import Model, Input
from keras.layers import LSTM, Dense, concatenate, Flatten
from keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.python.keras.layers.embeddings import Embedding

from .datagen_config import *
from .datagen import *

import tensorflow as tf





"""Generates model from datagen"""
def buildModelFromDatagen(datagen, embeddingDimension = 16):
    embedRanges = [len(cat) for cat in datagen.oe.categories]
    embeddings = [Embedding(dim, embeddingDimension)  for dim in embedRanges]
    outputDim = len(datagen.ohe.categories[0])
    dt = datagen.dt
    testSampleX, testSampleY = datagen.getBatches([dt])
    inputs = [Input(shape = (inp.shape[1])) for inp in testSampleX]
    embeddingOut = [embeddings[i%6](inputs[i]) for i,inp in enumerate(inputs)]

    leftLSTMInp = concatenate(embeddingOut[:6])
    middleLSTMInp = concatenate(embeddingOut[6:12])
    rightLSTMInp = concatenate(embeddingOut[12:18])

    leftLSTMOut1 = LSTM(200, return_sequences=True)(leftLSTMInp)
    leftLSTMOut2 = Flatten()(LSTM(200)(leftLSTMOut1))

    middleOut = Flatten()(Dense(200, activation = "relu")(middleLSTMInp))

    rightLSTMOut1 = LSTM(200, return_sequences=True)(rightLSTMInp)
    rightLSTMOut2 = Flatten()(LSTM(200)(rightLSTMOut1))

    concatenated = concatenate([leftLSTMOut2, middleOut, rightLSTMOut2])

    output = Dense(outputDim, activation="softmax")(concatenated)

    model = Model(inputs, output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model


"""Build model using preset config"""
def buildPresetModelAndDatagen(name):
    _datagen = Datagen.loadFromPreset(name)
    model = buildModelFromDatagen(_datagen)
    return model, _datagen




def trainPresetModel(name, batchsize = 250, fitConfig = {}):
    model, _datagen = buildPresetModelAndDatagen(name)
    trainDatagen = _datagen.getTrainBatches(batchsize)
    testDatagen = _datagen.getTestBatches(batchsize)
    model.fit(trainDatagen, validation_data = testDatagen, **fitConfig)
    return model, _datagen







