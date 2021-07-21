import music21
from music21 import stream, note
import os
import pickle
import pandas
from copy import deepcopy
from tqdm import tqdm
from itertools import compress
import pandas as pd

#############
#  CLASSES  #
#############

class Piece:
    """
    Encoded piece data structure
        Used for easier data generation
    """
    def __init__(self, soprano, alto, tenor, bass, title):
        self.title = title
        self.voices = [soprano, alto, tenor, bass]
        self.pad()
        self.metronome = self.getMetronome()
    
    def getMetronome(self):
        return [i%4 for i in range(len(self.alto))]
    
    def updateVoices(self):
        self.soprano = self.voices[0]
        self.alto = self.voices[1]
        self.tenor = self.voices[2]
        self.bass = self.voices[3]
        
    def pad(self):
        maxLen = max([len(voice) for voice in self.voices])
        self.voices = [voice + ["_" for i in range(maxLen - len(voice))] for voice in self.voices]
        self.updateVoices()
    
    def __repr__(self):
        return f'Soprano: {self.soprano} \n\nAlto: {self.alto} \n\nTenor: {self.tenor} \n\nBass: {self.bass} \n\nMetronome: {self.metronome}'
        



class EncodedPieces:
    def __init__(self, pieces):
        self.pieces = pieces

    def save(self, fp):
        df = self.toDf()
        df.to_csv(fp, index = False)

    def toDf(self):
        return pandas.DataFrame(
            data = 
                [
                    [
                        piece.soprano[i],
                        piece.alto[i],
                        piece.tenor[i],
                        piece.bass[i],
                        piece.metronome[i],
                        pieceId,
                        piece.title
                    ] for pieceId,piece in enumerate(self.pieces) for i in range(len(piece.soprano))
                ],
            columns = ["soprano", "alto", "tenor", "bass", "metronome", "pieceID", "title"]
        )




###############
#  FUNCTIONS  #
###############



def loadParsedMidis(_path):
    """
    Loads pickled list of music21 bach pieces
    _path: str
        Path of midis
    """
    f = open(_path, "rb")
    parsedMidis = pickle.load(f)
    return parsedMidis

def filterValidMidis(pieces):
    """
    Only return bach pieces are valid

    """
    condition = [set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in midi.parts]) for midi in pieces]
    parsedMidis = list(compress(pieces, condition))
    return parsedMidis

def addTranspositions(pieces):

    transposedPieces = []
    for piece in tqdm(pieces):
        transposedPieces.append(piece)
        for i in range(1,6):
            transposed = deepcopy(piece)
            for _note in transposed.recurse().getElementsByClass(note.Note):
                _note.pitch = _note.pitch.transpose(i)
            transposedPieces.append(transposed)
        for i in range(1,7):
            transposed = deepcopy(piece)
            for _note in transposed.recurse().getElementsByClass(note.Note):
                _note.pitch = _note.pitch.transpose(-i)
            transposedPieces.append(transposed)
            
    return transposedPieces

def tokenizePart(part):
    notes = []
    measures = part.getElementsByClass(stream.Measure)    
    for measure in measures:
        _notes = measure.getElementsByClass(note.Note)
        for _note in _notes:
            holds = ["_" for i in range(int(_note.duration.quarterLength//0.25) - 1)]
            notes.extend([_note.pitch.midi if "Note" in _note.classes else "_"]+holds)
    return notes

def tokenizePiece(piece):
    parts = {}
    title = piece.metadata.title
    for part in piece.parts:
        partID = part.id.lower()
        tokenized = tokenizePart(part)
        parts[partID] = tokenized
    return Piece(title = title,**parts)



def tokenizeManyPieces(pieces):
    return [tokenizePiece(piece) for piece in tqdm(pieces)]








##########
#  MAIN  #
##########

def encodePieces(_path):
    parsedMidis = loadParsedMidis(_path)
    filteredMidis = filterValidMidis(parsedMidis)
    print("Adding transpositions: ")
    transposed = addTranspositions(filteredMidis)
    print("Tokenizing pieces: ")
    encodedPieces = EncodedPieces(tokenizeManyPieces(transposed))
    encodedPieces.save("data/tokenize_transposed_pieces.csv")

def trainTestSplit(encodedPath, trainToTestRatio):
    df = pd.read_csv(encodedPath)
    trainDf = pd.DataFrame(columns = ["soprano", "alto", "tenor", "bass", "metronome", "pieceID"])
    testDf = pd.DataFrame(columns = ["soprano", "alto", "tenor", "bass", "metronome", "pieceID"])
    pieces = [group for i, group in tqdm(df.groupby("pieceID"))] 
    for i in pieces[:int((1-trainToTestRatio)*len(pieces))]:
        trainDf =trainDf.append(i)
    for i in pieces[int((1-trainToTestRatio)*len(pieces)):]:
        testDf = testDf.append(i)

    testDf.to_csv("data/test_df.csv", index=False)
    trainDf.to_csv("data/train_df.csv", index=False)
    

def main():
    _path = "data/parsed_midis.pickle"
    encodePieces(_path)
    trainTestSplit("data/tokenize_transposed_pieces.csv", 0.15)
    



if __name__=="__main__":
    main()





