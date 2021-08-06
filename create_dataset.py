import random
import music21

from src.dataset.dataset import *
from src.dataset.meta_manager import *

chorales = music21.corpus.chorales.Iterator

def getData():
    paths = music21.corpus.getComposer('bach')
    parsed = []
    for path in tqdm(paths):
        piece = music21.corpus.parse(path)
        if(set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in piece.parts]) and getTimeSignature(piece)==["4/4"]):
            parsed.append(piece)
    random.shuffle(parsed)
    return parsed


def getTestData():
    paths = music21.corpus.getComposer('bach')
    parsed = []
    for path in tqdm(paths[:10]):
        piece = music21.corpus.parse(path)
        if(set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in piece.parts]) and getTimeSignature(piece)==["4/4"]):
            parsed.append(piece)
    random.shuffle(parsed)
    return parsed

dataset = BachChorale("bach_default", getData(), [BeatMeta(), FermataMeta(), KeyMeta()], 16, overwrite=False)
dataset.create_dataset()
print(dataset.get_datagen(10000, 0.1)[0])
print(dataset.get_datagen(10000, 0.1)[1])
print(dataset.get_datagen(10000, 0.1)[2])
print(dataset.get_datagen(10000, 0.1)[3])



