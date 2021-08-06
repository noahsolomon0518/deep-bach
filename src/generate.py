from operator import index
from pickle import load
from mido import MidiFile, MidiTrack, Message
import numpy as np
from keras.models import load_model
from tqdm import tqdm
from pathlib import Path
import sys
import os









dtInterval = 130
def save_as_midi(chorale_tensor, fp):
    mf = MidiFile()
    tracks = [MidiTrack() for i in range(4)]
    mf.tracks = tracks
    for i, track in enumerate(tracks):
        for msgInd, msg in enumerate(chorale_tensor[:,i]):
            if(msg!="_"):
                track.append(Message("note_on", note = int(msg), time = 0))
                dt = 0
                c = 1
                while(msgInd+c < len(chorale_tensor[:,i])-1 and chorale_tensor[msgInd+c,i]=="_"):
                    dt+=dtInterval
                    c+=1
                dt+=dtInterval
                track.append(Message("note_off", note = int(msg), time = dt))
    mf.save(fp)



def sample(distributions, temperature = 1):
    preds = np.log(distributions)/temperature
    exp = np.exp(preds)
    sampleSum = np.sum(exp, axis = 1)
    expSum = np.vstack([e/sampleSum[i] for i, e in enumerate(exp)])
    expSum = np.array(expSum).astype('float64')
    expSum = expSum/(np.sum(expSum, 1).reshape(-1,1))
    probs = np.vstack([np.random.multinomial(1, e, 1) for e in expSum])
    return np.argmax(probs, axis = 1)


def getNoteDistributions(datagen):

    voices = np.transpose(np.array(datagen.data)[:,:4])
    p = []
    length = len(voices[0,:])
    for i,voice in enumerate(voices):

        p.append([len([note for note in voice if note == uniqueNote])/length for uniqueNote in datagen.oe.categories[i]])
    return p


def generate(models, dataset, iterations = 10, length = 100, batch_size = 16):
    generated = dataset.intialize_random_chorale(length)
    sequence_length = dataset.sequence_length
    for i in tqdm(range(iterations)):
        for voice_id in range(4):
            datagen = dataset.get_datagen(voice_id)
            chorale_generation_inds = np.random.choice(range(0, length - (2*sequence_length+1)), size = batch_size, replace = False)
            samples = np.vstack([np.array([generated[ind:ind+2*sequence_length+1]]) for ind in chorale_generation_inds])
            features, _ = datagen.create_features_and_labels(samples, voice_id)
            predictions = models[voice_id].predict(features)
            index_predictions = sample(predictions)

            decoded_predictions = np.array([datagen.label_encoder.categories[0][index_pred] for index_pred in index_predictions])
            encoded_predictions = [datagen.voice_encoder.categories[voice_id].index(pred) for pred in decoded_predictions]

            generated[chorale_generation_inds+sequence_length, voice_id] = encoded_predictions
    return dataset.voice_encoder.inverse_transform(generated[sequence_length:len(generated)-(sequence_length+1),:4])





