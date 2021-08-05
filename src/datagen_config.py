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


"""Datasets"""
normalDf = "data/df/tokenized_normal.csv"
convertedToCDf = None


class ConfigLoader:
    def __init__(self, configName):
        self.config = config[configName]
        data = pd.read_csv(self.config["dataPath"])
        self.config["inputRanges"] = getPitchRanges(data, self.config["transpositionRange"]) + getBeatRange() + getFermataRange()
        self.config["outputRanges"] = getNetPitchRange(data, self.config["transpositionRange"])



config = {
    "default": {
        "dt": 16,
        "transpositionRange": (-6,6),
        "dataPath": normalDf
    },

    "more_dt": {
        "dt": 32,
        "transpositionRange": (-6,6),
        "dataPath": normalDf
    },
    
    "less_dt": {
        "dt": 4,
        "transpositionRange": (-6,6),
        "dataPath": normalDf
    },
    
    "more_transposition": {
        "dt": 16,
        "transpositionRange": (-12,12),
        "dataPath": normalDf
    },
    
    "no_transposition": {
        "dt": 16,
        "transpositionRange": (-0,0),
        "dataPath": normalDf
    }

}

