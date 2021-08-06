import music21
import numpy as np

"""Number of sixteenth notes in a chorale"""
def get_chorale_length(chorale):
    measures = chorale.parts[0].getElementsByClass(music21.stream.Measure)
    return int((measures[-1].offset+4)//0.25)

def get_n_sharps(chorale):
    return list(set([ks.sharps for ks in chorale.recurse().getKeySignatures()]))[0]


class MetaManager:

    def __init__(self, dimension):
        self.dimension = dimension


    def tokenize_meta(self, chorales):
        pass

    def transpose_tokenized_meta(self, tokenized_meta, semi_tones):
        pass

    def intialize_random(self, length):
        pass


class BeatMeta(MetaManager):

    def __init__(self):
        super().__init__(dimension = 4)


    def tokenize_meta(self, chorales):
        return [
            self.get_beats(chorale) for chorale in chorales
        ]

    def get_beats(self, chorale):
        return np.array([i%4 for i in range(get_chorale_length(chorale))]).reshape(-1,1)

    def transpose_tokenized_meta(self, tokenized_meta, semi_tones):
        return tokenized_meta
        
    def intialize_random(self, length):
        return np.array([i%4 for i in range(length)]).reshape(-1,1)



class FermataMeta(MetaManager):

    def __init__(self):
        super().__init__(dimension = 2)


    def tokenize_meta(self, chorales):
        return [
            self.get_fermatas(chorale) for chorale in chorales
        ]

    def get_fermatas(self, chorale):
        tokenized = [0 for i in range(get_chorale_length(chorale))]
        for note in chorale.recurse().getElementsByClass(music21.note.Note):
            if(any([type(element)==music21.expressions.Fermata for element in note.expressions])):
                tokenized[int((note.activeSite.offset+note.offset)//0.25)] = 1
        return np.array(tokenized).reshape(-1,1)

    def transpose_tokenized_meta(self, tokenized_meta, semi_tones):
        return tokenized_meta
    
    def intialize_random(self, length):
        return np.array([0 for i in range(length)]).reshape(-1,1)



class KeyMeta(MetaManager):

    SHARPS_TO_KEY_START = {
        0:0,
        1:7,
        2:2,
        3:9,
        4:4,
        5:11,
        6:6,
        7:1,
        -1:5,
        -2:10,
        -3:3,
        -4:8,
        -5:1,
        -6:6,
        -7:11
    }

    def __init__(self):
        super().__init__(dimension = 12)


    def tokenize_meta(self, chorales):
        return [
            self.get_key(chorale) for chorale in chorales
        ]

    def get_key(self, chorale):
        key = KeyMeta.SHARPS_TO_KEY_START[get_n_sharps(chorale)]
        return np.array([key for i in range(get_chorale_length(chorale))]).reshape(-1,1)

    def transpose_tokenized_meta(self, tokenized_meta, semi_tones):
        key = tokenized_meta[0][0]
        new_key = (key+semi_tones)%12
        
        return np.array([new_key for i in range(len(tokenized_meta))]).reshape(-1,1)
    

    def intialize_random(self, length):
        return np.array([0 for i in range(length)]).reshape(-1,1)







