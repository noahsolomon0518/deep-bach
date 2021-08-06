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


def generate(model, dataset, iterations = 10, length = 100, batch_size = 16):
    generated = dataset.intialize_random_chorale(length)
    datagen = dataset.datagen
    sequence_length = dataset.sequence_length
    for i in range(iterations):
        chorale_generation_inds = np.random.choice(range(0, length - (2*sequence_length+1)), size = batch_size, replace = False)
        voice_mask_inds = np.random.randint(0, 4, len(chorale_generation_inds))
        samples = np.vstack([np.array([generated[ind:ind+2*sequence_length+1]]) for ind in chorale_generation_inds])
        features, _ = datagen.create_features_and_labels(samples)
        predictions = model.predict(features)
        index_predictions = sample(predictions)

        decoded_predictions = [dataset.label_encoder.categories[0][index_pred] for index_pred in index_predictions]
        print(decoded_predictions)
        for i,voice_mask_ind in enumerate(voice_mask_inds):

            dataset.voice_encoder.categories[voice_mask_ind].index(decoded_predictions[i])
        generated[chorale_generation_inds, voice_mask_inds] = np.array([
            dataset.voice_encoder.categories[voice_mask_ind].index(decoded_predictions[i]) for i,voice_mask_ind in enumerate(voice_mask_inds)
        ])
    return dataset.voice_encoder.inverse_transform(generated[:,:4])





