from keras.models import load_model
import pickle
import time
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse
import pandas as pd
from mido import MidiFile, MidiTrack, Message
sys.path.append(os.path.join(os.getcwd(), "src\\models"))

from create_datagen import XYSampler, Datagen
MODEL_PATH = "models/"

dtInterval = 100
def dfPieceToMidi(dfPiece):
    mf = MidiFile()
    tracks = [MidiTrack() for i in range(4)]
    mf.tracks = tracks
    for i, track in enumerate(tracks):
        for msgInd, msg in enumerate(dfPiece.iloc[:,i]):
            if(msg!="_"):
                track.append(Message("note_on", note = int(msg), time = 0))
                dt = 0
                c = 1
                while(msgInd+c != len(dfPiece.iloc[:,i])-1 and dfPiece.iloc[msgInd+c,i]=="_"):
                    dt+=dtInterval
                    c+=1
                dt+=dtInterval
                track.append(Message("note_off", note = int(msg), time = dt))
               
                
                
    return mf

def sample(unencoded, xYSampler, model):
    past, current, future = xYSampler.encodeX(unencoded, xYSampler.dt)
    x = \
    [np.array([x]) for x in np.transpose(past)] + \
    [np.array([[x]]) for x in np.transpose(current)] + \
    [np.array([x]) for x in np.transpose(future)]

    pred = model.predict(x)
    argmax = np.argmax(pred[0])
    return xYSampler.ohe.categories[0][argmax]

def sampleMany(unencoded, xYSampler, model):
    x = [[] for i in range(15)]
    for unenc in unencoded:
        past, current, future = xYSampler.encodeX(unenc.iloc[:,:5], xYSampler.dt)
        curX = \
        [np.array(x) for x in np.transpose(past)] + \
        [np.array([x]) for x in np.transpose(current)] + \
        [np.array(x) for x in np.transpose(future)]
        for i in range(15):
            x[i].append(curX[i])
    x = [np.stack(i) for i in x]
    pred = model.predict(x)
    argmaxs = [np.argmax(p) for p in pred]
    return [xYSampler.ohe.categories[0][argmax] for argmax in argmaxs]
    



def generateFromPiece(dfPiece, dt, xYSampler, model, iterations = 10):
    indices = [list(range(dt+i, len(dfPiece.iloc[:,0])-(dt+1), dt+1)) for i in range(dt)]
    for i in tqdm(range(iterations)):
        for d in range(dt):
            sampleInds = indices[d]
            voiceInds = np.random.choice(range(4), len(sampleInds))
            temp = dfPiece.to_numpy()
            temp[sampleInds, voiceInds] = "x"
            dfPiece.iloc[:,:] = temp
            samples = [dfPiece.iloc[sampleInds[ind]-dt:sampleInds[ind]+dt+1,:] for ind in range(len(sampleInds))]
            sampled = sampleMany(samples, xYSampler, model)
            arr = dfPiece.to_numpy()
            arr[sampleInds, voiceInds] = sampled
            dfPiece.iloc[:,:] = arr
    return dfPiece



def generateMusic(modelPath, piece, savePath = None, iterations = 10):
    dgp = open(MODEL_PATH+modelPath+"/train_datagen.pickle", "rb")
    
    model = load_model(MODEL_PATH+modelPath+"/model.h5")
    datagen = pickle.load(dgp)
    sampler = datagen.sampler

    generated = dfPieceToMidi(generateFromPiece(piece, sampler.dt, sampler, model, iterations = iterations))

    if(savePath == None):
        generated.save(str(time.time())+".mid")    
    else:
        generated.save(savePath)

    print("Generated Music Saved.")









