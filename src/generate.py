from pickle import load
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

dtInterval = 130




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



"""Provides tools for intializing pieces and generating new pieces"""
class AIComposer:
    def __init__(self, piece, model, datagen) -> None:
        self.piece = piece
        self.datagen = datagen
        self.model = model
        print(datagen.oe.categories)
    
    """Intialize piece by name from bach chorales. Requires datagen object"""
    @staticmethod
    def initPiece(name, model, datagen, ext = ".mxl", pickBy = "pieceName"):
        data = datagen.data
        return AIComposer(data[data[pickBy]==name+ext], model, datagen)


    """Intialize piece by name from bach chorales. Requires datagen name"""
    @staticmethod
    def initPieceFromPreset(name, preset, *args, **kwargs):
        datagen = Datagen.loadFromPreset(preset)
        model = load_model("models/"+preset+"/model.h5")
        return AIComposer.initPiece(name, model, datagen, *args, **kwargs)

    """Intialize randomly from datagen object"""
    @staticmethod
    def initRandomly(model, datagen, length = 100, padding = None):
        data = []
        beats = [i%4 for i in range(length)]
        fermatas = [0 for i in range(length)]

        #will pick random key

        categories = datagen.oe.categories
        
        print(categories)
        ps = getNoteDistributions(datagen)

        for i in range(4):
            data.append([np.random.choice(categories[i][:-1], p = ps[i][:-1]) for l in range(length)])
        
        data.append(beats)
        data.append(fermatas)
        data = np.transpose(np.array(data))
        if(type(padding) != type(None)):
            padding = np.array(padding)
            data[:datagen.dt,:5] = padding[:datagen.dt,:5]
            data[-datagen.dt:,:5] = padding[-datagen.dt:,:5]

        return AIComposer(data, model, datagen)

    """Intialize randomly from datagen preset name"""
    @staticmethod
    def initRandomlyFromPreset(preset, *args, **kwargs):
        datagen = Datagen.loadFromPreset(preset)
        model = load_model("models/"+preset+"/model.h5")
        return AIComposer.initRandomly(model, datagen, *args, **kwargs)
        


    @property
    def midi(self):
        piece = np.array(self.piece)
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

    """Takes existing piece and adds noise (random notes) to piece within range"""
    def noisify(self, noiseRange, probability = 1.0):
        piece = np.array(self.piece)
        datagen = self.datagen
        assert noiseRange[0]>=datagen.dt
        assert noiseRange[1]<=len(piece)-datagen.dt
        nToGenerate = noiseRange[1] - noiseRange[0]
        noteDistribution = getNoteDistributions(datagen)
        noisified = np.array([
            np.random.choice(datagen.oe.categories[i], size = nToGenerate, p = noteDistribution[i]) for i in range(4)
        ])

        piece[noiseRange[0]:noiseRange[1], :4] = np.transpose(noisified)
        self.piece = piece


    def gibbsSample(self, generationRange = None, voiceRange = range(0,4), iterations = 100, temperature = 1, sample = False):
        dt = self.datagen.dt
        if generationRange == None:
            generationRange = (dt, len(self.piece)-dt+1)
        datagen = self.datagen

        generated = self.piece
        model = self.model
        print(f"Generating piece with: \n   Iterations = {iterations} \n   temperature = {temperature}")
        for i in tqdm(range(iterations)):
            start = np.random.randint(generationRange[0], generationRange[0]+dt)
            sampleInds = list(range(start, generationRange[1], 2*dt+1))
            voiceInds = np.random.choice(list(voiceRange), len(sampleInds))
            generated[sampleInds, voiceInds] = "x"
            samples = encodeXSamples(np.vstack([np.array(generated[sampleInds[ind]-dt:sampleInds[ind]+dt+1,:6]) for ind in range(len(sampleInds))]), dt, datagen.oe)
            predictions = sample(model.predict(samples)) if sample else np.argmax(model.predict(samples), axis = 1)
            predictions = np.array([datagen.ohe.categories[0][prediction] for prediction in predictions])
            for predInd,prediction in enumerate(predictions):
                if(prediction not in datagen.oe.categories[voiceInds[predInd]]):
                    predictions[predInd] = "_"
            generated[sampleInds, voiceInds] = predictions
        self.piece = generated
        
    def save(self, fp):
        self.midi.save(fp)


        



    




"""


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



"""
