from mido import MidiFile, MidiTrack, Message
import numpy as np
from keras.models import load_model
from tqdm import tqdm
from pathlib import Path
import sys
import os



from tensorflow.python.keras.backend import argmax, dtype

from .datagen_config import *
from .datagen import *



"""For saving a piece in the form of a dataframe to midi format"""
dtInterval = 180
def pieceToMidi(piece):
    piece = np.array(piece)
    mf = MidiFile()
    tracks = [MidiTrack() for i in range(4)]
    mf.tracks = tracks
    for i, track in enumerate(tracks):
        for msgInd, msg in enumerate(piece[:,i]):
            if(msg not in ["_","x"]):
                track.append(Message("note_on", note = int(msg), time = 0))
                dt = 0
                c = 1
                while(msgInd+c < len(piece[:,i])-1 and piece[msgInd+c,i]=="_"):
                    dt+=dtInterval
                    c+=1
                dt+=dtInterval
                track.append(Message("note_off", note = int(msg), time = dt))  
    return mf

"""Returns piece(s) in the form a dataframe by name"""
def getPieces(pieceNames, dfPath = "data/df/tokenized_normal.csv", pickBy = "pieceName", ext = ".mxl"):
    df = pd.read_csv(dfPath)
    dfPieces = []
    if(type(pieceNames)!=list):
        pieceNames = [pieceNames]
    for pieceName in pieceNames:
        dfPieces.append(df[df[pickBy]==pieceName+ext])

    if(len(dfPieces)==1):
        return dfPieces[0]
    return dfPieces
    

def intializeRandomPiece(length, datagen, padding = None, distribution = None):
    padding = np.array(padding)
    data = []
    beats = [i%4 for i in range(length)]
    fermatas = [0 for i in range(length)]

    categories = datagen.oe.categories
    for cat in categories[:4]:
        cat.remove("x")

    ps = getNoteDistributions(datagen)

    for i in range(4):
        data.append([np.random.choice(categories[i], p = ps[i]) for l in range(length)])
    
    data.append(beats)
    data.append(fermatas)
    data = np.transpose(np.array(data))
    if(type(padding) != None):
        padding = np.array(padding)
        data[:datagen.dt,:5] = padding[:datagen.dt,:5]
        data[-datagen.dt:,:5] = padding[-datagen.dt:,:5]

    return data


def intializeRandomPieceFromPreset(preset, length, padding = None, distribution = None):
    _datagen = Datagen.loadFromPreset(preset)
    return intializeRandomPiece(length, _datagen, padding, distribution)




def sample(distributions, temperature = 1):
    preds = np.log(distributions)/temperature
    exp = np.exp(preds)
    sampleSum = np.sum(exp, axis = 1)
    expSum = np.vstack([e/sampleSum[i] for i, e in enumerate(exp)])
    probs = np.vstack([np.random.multinomial(1, e, 1) for e in expSum])
    return np.argmax(probs, axis = 1)


def getNoteDistributions(datagen):

    voices = np.transpose(np.array(datagen.data)[:,:4])
    p = []
    length = len(voices[0,:])
    for i,voice in enumerate(voices):

        p.append([len([note for note in voice if note == uniqueNote])/length for uniqueNote in datagen.oe.categories[i]])
    return p


def generateFromPiece(piece, model, datagen, iterations = 20, temperature = 1, sample = False):
    dt = datagen.dt
    generated = np.array(piece)[:,:6]
    indices = [list(range(dt+i, len(generated[:,0])-(dt+1), dt+1)) for i in range(dt)]
    print(f"Generating piece with: \n   Iterations = {iterations} \n   temperature = {temperature}")
    for i in tqdm(range(iterations)):
        for d in range(dt):
            sampleInds = indices[d]
            voiceInds = np.random.choice(range(4), len(sampleInds))
            generated[sampleInds, voiceInds] = "x"
            samples = encodeXSamples(np.vstack([np.array(generated[sampleInds[ind]-dt:sampleInds[ind]+dt+1,:6]) for ind in range(len(sampleInds))]), dt, datagen.oe)
            predictions = sample(model.predict(samples)) if sample else np.argmax(model.predict(samples), axis = 1)
            predictions = np.array([datagen.ohe.categories[0][prediction] for prediction in predictions])
            generated[sampleInds, voiceInds] = predictions
    return generated




def generateFromModel(modelName, piece = None, iterations = 20, temperature = 1, savePath = None, sample = False):
    model = load_model("models/"+modelName+"/model.h5")
    datagen = Datagen.load("models/"+modelName+"/datagen.dg")
    generated = generateFromPiece(piece, model, datagen, iterations, temperature, sample = sample)
    if(savePath != None):
        Path("models/"+ modelName+"/generated").mkdir(parents=True, exist_ok=True)
        pieceToMidi(piece).save("models/"+ modelName+"/generated/"+savePath)
    return generated








