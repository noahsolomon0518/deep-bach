from itertools import chain
import pandas as pd

"""Get pitch range for one column"""
def getPitchRange(col, transpositionRange = (0,0)):
    col = col[lambda x:x!="_"]
    ranges = list(range(int(col.min())+transpositionRange[0], int(col.max())+transpositionRange[1]+1))
    return sorted(list([str(x) for x in ranges] + ["_", "x"]))

"""Get pitch range for all columns"""
def getPitchRanges(df, transpositionRange = (0,0)):
    return [getPitchRange(df.iloc[:, i], transpositionRange) for i in range(4)]


def getBeatRange(n=4):
    return [[i for i in range(n)]]

def getFermataRange():
    return [[0,1]]


"""Used for one hot encoding"""
def getNetPitchRange(df, transpositionRange = (0,0)):
    pitches = [list(sorted(set(chain.from_iterable(getPitchRanges(df, transpositionRange)))))]
    pitches[0].remove("x")
    return pitches

class ConfigLoader:
    def __init__(self, dataPath, dt, transpositionRange):
        self.dataPath = dataPath
        self.dt = dt
        self.transpositionRange = transpositionRange
        data = pd.read_csv(dataPath)
        self.inputRange = getPitchRanges(data, transpositionRange) + getBeatRange() + getFermataRange()
        self.outputRange = getNetPitchRange(data, transpositionRange)



dtConfig = {
    "default": {
        "dt": 16,
        "transpositionRange": (-12,12),
        "dataPath": "data/df/tokenized_normal.csv"
    },

    "more_dt": {
        "dt": 32,
        "transpositionRange": (-12,12),
        "dataPath": "data/df/tokenized_normal.csv"
    },
    
    "less_dt": {
        "dt": 4,
        "transpositionRange": (-12,12),
        "dataPath": "data/df/tokenized_normal.csv"
    }
}

