
import numpy as np
from numpy.lib.arraysetops import unique
from itertools import chain
import music21
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from .meta_manager import *
import pickle
import os
from keras.utils import Sequence


def getTimeSignature(piece):
    return list(set([ts.ratioString for ts in piece.recurse().getTimeSignatures()]))


"""Creates dataset based on voice manager and meta manager"""
class BachChorale:
    def __init__(self, name, chorales_func, meta_managers, sequence_length, batch_size = 10000, test_size = 0.1, transpose = True, overwrite = False):
        self.name = name
        self.batch_size = batch_size
        self.test_size = test_size
        self.overwrite = overwrite
        self.chorales = [chorale for chorale in chorales_func()]
        self.meta_managers = meta_managers
        self.sequence_length = sequence_length
        self.transpose = transpose
        self.data = None

    """Creates entire dataset with transpositions"""
    def create_dataset(self):
        #calculates voice ranges and intializes ordinal encoder
        chorales = self.chorales
        data = []
        tokenized_chorales = self.tokenize_chorales(chorales)
        tokenized_metas = [mm.tokenize_meta(chorales) for mm in self.meta_managers]
        self.voice_ranges = self.calculate_voice_ranges(tokenized_chorales)
        self.tokenized_voice_categories = self.calculate_tokenized_voice_categories(tokenized_chorales)
        self.voice_encoder = OrdinalEncoder(categories=self.tokenized_voice_categories)
        self.label_encoder = self.get_label_encoder()
        self.voice_encoder.fit([[self.voice_encoder.categories[encoder_num][0] for encoder_num in range(4)]])
        self.label_encoder.fit([[self.label_encoder.categories[0][0]]])
        self.get_possible_transpositions(tokenized_chorales[0], 16, 32)
        if(os.path.exists("data/datasets/"+self.name) and self.overwrite == False):
            print("Loading dataset from cache")
            with open("data/datasets/"+self.name, "rb") as f:
                self.data = pickle.load(f)
                return self.data
        print("Building dataset")
        for choraleID in tqdm(range(len(tokenized_chorales))):
            tokenized_chorale = tokenized_chorales[choraleID]
            chorale_meta = [tokenized_meta[choraleID] for tokenized_meta in tokenized_metas]
            chorale_transpositions = {}
            meta_transpositions = {}
            for start_ind in range(0, len(tokenized_chorale)-(2*self.sequence_length+1)):
                #end_ind is exclusive
                end_ind = start_ind+2*self.sequence_length+1
                transposition_intervals = self.get_possible_transpositions(tokenized_chorale, start_ind, end_ind)
                for semi_tone in transposition_intervals:
                    if(semi_tone not in chorale_transpositions.keys()):
                        chorale_transpositions[semi_tone] = self.transpose_tokenized_chorale(tokenized_chorale, semi_tone)
                        meta_transpositions[semi_tone] = [self.meta_managers[i].transpose_tokenized_meta(meta, semi_tone) for i, meta in enumerate(chorale_meta)]
                    voice_tensor = self.voice_encoder.fit_transform(chorale_transpositions[semi_tone][start_ind:end_ind])
                    meta_tensors = np.hstack([meta[start_ind:end_ind] for meta in meta_transpositions[semi_tone]])
                    sample_tensor = np.hstack([voice_tensor, meta_tensors])
                    data.append(np.expand_dims(sample_tensor, 0))
        data = np.vstack(data)
        with open("data/datasets/"+self.name, "wb") as f:
            pickle.dump(data, f)
        self.data = data
        return data
        

    
    """Tokenizes each voice as actual notes"""
    def tokenize_chorales(self, chorales):
        return [
            np.hstack([self.tokenize_part(chorale, partID).reshape(-1,1) for partID in ["Soprano", "Alto", "Tenor", "Bass"]]) for chorale in chorales
        ]

    """Tokenize single part of music21 chorale"""
    def tokenize_part(self, chorale, partID):
        part = [part for part in chorale.parts if part.id == partID][0]
        measures = part.getElementsByClass(music21.stream.Measure)
        tokenized = ["_" for i in range(int((measures[-1].offset+4)//0.25))]
        for measure in measures:
            measureOffset = measure.offset
            notes = measure.getElementsByClass(music21.note.Note)
            for note in notes:
                tokenized[int((measureOffset+note.offset)//0.25)] = note.pitch.midi
        return np.array(tokenized)


    """Calculates min and max midi range of tokenized_chorales"""
    def calculate_voice_ranges(self, tokenized_chorales):
        if(type(tokenized_chorales)!=list):
            tokenized_chorales = [tokenized_chorales]
        return [
            (int(min(unique_midi_notes)), int(max(unique_midi_notes))) for unique_midi_notes in [
                set(chain.from_iterable([
                    tokenized_chorale[tokenized_chorale[:,partID]!="_",partID] for tokenized_chorale in tokenized_chorales
                ])) for partID in range(4)
            ]
        ]
    
    """Calculates voice ranges for ordinal encoder"""
    def calculate_tokenized_voice_categories(self, tokenized_chorales):
        if(type(tokenized_chorales)!=list):
            tokenized_chorales = [tokenized_chorales]

        return [
            sorted([str(i) for i in list(range(int(min(unique_notes)), int(max(unique_notes))+1)) + ["x", "_"]]) for unique_notes in [
                set(chain.from_iterable([
                    tokenized_chorale[tokenized_chorale[:,partID]!="_",partID] for tokenized_chorale in tokenized_chorales
                ])) for partID in range(4)
            ]
        ]

    """Calculates all possible transpositions that a sample can make adhering to voice ranges"""
    def get_possible_transpositions(self, tokenized_chorale, start_ind, end_ind):
        sample_ranges = self.calculate_voice_ranges(tokenized_chorale[start_ind:end_ind])
        range_differences = [
            (voice_min-sample_min, voice_max - sample_max) for (sample_min, sample_max) , (voice_min, voice_max) in zip(sample_ranges, self.voice_ranges)
        ]
        return range(max(np.array(range_differences)[:,0]), min(np.array(range_differences)[:,1])+1)

    def transpose_tokenized_chorale(self, chorale, interval):
        return np.vstack([
            np.array([str(int(note) + interval) if note != "_" else "_" for note in timestep]) for timestep in chorale
        ])

    def get_label_encoder(self):
        min_note, max_note = min(np.array(self.voice_ranges)[:,0]), max(np.array(self.voice_ranges)[:,1])
        ohe = OneHotEncoder(categories=[[str(i) for i in range(min_note, max_note+1)] + ["_"]], sparse=False)
        return ohe

    def intialize_random_chorale(self,length=100):
        return np.hstack([self.intialize_random_voices(length)]+[mm.intialize_random(length) for mm in self.meta_managers])

    def intialize_random_voices(self, length):
        return np.hstack([np.random.choice(len(self.voice_encoder.categories[voiceID])-1, length).reshape(-1,1) for voiceID in range(4)])

    @property
    def datagen(self):
        return Datagen(self.data, self.voice_encoder, self.label_encoder, self.sequence_length, self.batch_size, self.test_size)



class Datagen(Sequence):
    def __init__(self, data, voice_encoder, output_encoder, sequence_size, batchsize = 10000, test_size = 0.1): 
        self.train_data, self.test_data = data[:round(len(data)*(1-test_size))], data[round(len(data)*(1-test_size)):]
        self.batchsize = batchsize
        self.output_encoder = output_encoder
        self.voice_encoder = voice_encoder
        self.sequence_size = sequence_size
    
    def on_epoch_end(self):
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)

    def create_features_and_labels(self, data, voice_mask_inds = None):
        if(voice_mask_inds==None):
            voice_masks_inds = np.random.randint(0,4, len(data))
        data = np.copy(data)
        masked_samples, labels = self.mask_samples(data, voice_masks_inds)
        reversed_samples = self.reverse_samples(masked_samples)
        return [
            reversed_samples[:,ranges,column_ind].reshape(len(data), len(ranges)) for ranges in [range(0,self.sequence_size), [self.sequence_size], range(self.sequence_size+1, 2*self.sequence_size+1)] for column_ind in range(len(reversed_samples[0][0]))
        ], labels

    def __getitem__(self, index):
        return self.create_features_and_labels(self.train_data[index*self.batchsize:(index+1)*self.batchsize])

    @property
    def test_dataset(self):
        return self.create_features_and_labels(self.test_data)


    def mask_samples(self, data, mask_inds):
        labels = []
        for i, sample in enumerate(data):
            
            yEncoded = data[i, self.sequence_size, mask_inds[i]]
            yDecoded = self.voice_encoder.categories[mask_inds[i]][int(yEncoded)]
            labels.append(yDecoded)

            data[i, self.sequence_size, mask_inds[i]] = self.voice_encoder.categories[mask_inds[i]].index("x")
        return data, self.output_encoder.fit_transform(np.array(labels).reshape(-1,1))
        

    def reverse_samples(self, samples):
        for i in range(len(samples)):
            samples[i, self.sequence_size+1:, :] = np.flipud(samples[i, self.sequence_size+1:, :])
        return samples



    def __len__(self):
        return len(self.train_data)//self.batchsize


    
        
