import random
from music21 import note, stream
import music21
from music21.stream import Score
import pickle
from pandas.core.frame import DataFrame
from tqdm import tqdm
import pandas as pd

"""Script that converts music21 pieces to dataframe"""

def getTimeSignature(piece):
    return list(set([ts.ratioString for ts in piece.recurse().getTimeSignatures()]))

def getKeySignature(piece):
    return list(set([ts.ratioString for ts in piece.recurse().getKeySignatures()]))

def getData():
    paths = music21.corpus.getComposer('bach')
    parsed = []
    for path in tqdm(paths):
        piece = music21.corpus.parse(path)
        if(set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in piece.parts]) and getTimeSignature(piece)==["4/4"]):
            parsed.append(piece)
    random.shuffle(parsed)
    return parsed

"""Number of sixteenth notes in a piece"""
def getPieceLength(piece):
    measures = piece.parts[0].getElementsByClass(music21.stream.Measure)
    return int((measures[-1].offset+4)//0.25)

def tokenizePart(piece, partID):
    part = [part for part in piece.parts if part.id == partID][0]
    measures = part.getElementsByClass(music21.stream.Measure)
    tokenized = ["_" for i in range(int((measures[-1].offset+4)//0.25))]
    for measure in measures:
        measureOffset = measure.offset
        notes = measure.getElementsByClass(music21.note.Note)
        for note in notes:
            tokenized[int((measureOffset+note.offset)//0.25)] = note.pitch.midi
    return tokenized

def tokenizeKey(piece):
    key = getKeySignature(piece)
    return [key for i in range(getPieceLength(piece))]

def tokenizePieceID(piece, n):
    return [n for i in range(getPieceLength(piece))]

def tokenizePieceName(piece):
    return [piece.metadata.title for i in range(getPieceLength(piece))]

def tokenizeBeats(piece, n):
    return [i%n for i in range(getPieceLength(piece))]

def tokenizeFermatas(piece):
    tokenized = [0 for i in range(getPieceLength(piece))]
    for note in piece.recurse().getElementsByClass(music21.note.Note):
        if(any([type(element)==music21.expressions.Fermata for element in note.expressions])):
            tokenized[int((note.activeSite.offset+note.offset)//0.25)] = 1
    return tokenized


"""Preprocesses music21 pieces"""
def preprocessNormal(pieces):
    data = {
        "soprano": [], 
        "alto": [],
        "tenor": [],
        "bass": [],
        "beat": [],
        "fermatas": [],
        "pieceID": [],
        "pieceName": []
        
    }
    for i,piece in enumerate(pieces):
        data["soprano"].extend(tokenizePart(piece, "Soprano"))
        data["alto"].extend(tokenizePart(piece, "Alto"))
        data["tenor"].extend(tokenizePart(piece, "Tenor"))
        data["bass"].extend(tokenizePart(piece, "Bass"))
        data["beat"].extend(tokenizeBeats(piece, 4))
        data["pieceID"].extend(tokenizePieceID(piece, i))
        data["fermatas"].extend(tokenizeFermatas(piece))
        data["pieceName"].extend(tokenizePieceName(piece))
    df = DataFrame(data)
    return df



preprocessNormal(getData()).to_csv("data/df/tokenized_normal.csv", index=False)

