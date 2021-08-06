from keras.models import Model, Input
from keras.layers import LSTM, Dense, concatenate, Flatten
from keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.python.keras.layers.embeddings import Embedding


import tensorflow as tf

"""Build models from dataset"""
def build_model(dataset, embedding_dimension):
    datagen = dataset.datagen
    embed_ranges = [len(cat) for cat in dataset.datagen.voice_encoder.categories] + [mm.dimension for mm in dataset.meta_managers]
    n_columns = len(embed_ranges)
    embeddings = [Embedding(dim, embedding_dimension)  for dim in embed_ranges]
    output_dimension = len(datagen.output_encoder.categories[0])
    testSampleX = datagen[0][0]
    inputs = [Input(shape = (inp.shape[1])) for inp in testSampleX]
    embeddingOut = [embeddings[i%n_columns](inputs[i]) for i,inp in enumerate(inputs)]

    leftLSTMInp = concatenate(embeddingOut[:n_columns])
    middleLSTMInp = concatenate(embeddingOut[n_columns:2*n_columns])
    rightLSTMInp = concatenate(embeddingOut[2*n_columns:3*n_columns])

    leftLSTMOut1 = LSTM(200, return_sequences=True)(leftLSTMInp)
    leftLSTMOut2 = Flatten()(LSTM(200)(leftLSTMOut1))

    middleOut = Flatten()(Dense(200, activation = "relu")(middleLSTMInp))

    rightLSTMOut1 = LSTM(200, return_sequences=True)(rightLSTMInp)
    rightLSTMOut2 = Flatten()(LSTM(200)(rightLSTMOut1))

    concatenated = concatenate([leftLSTMOut2, middleOut, rightLSTMOut2])

    output = Dense(output_dimension, activation="softmax")(concatenated)

    model = Model(inputs, output)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model












