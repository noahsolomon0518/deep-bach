from src.generate import generateFromModel
from src.generate import *




piece = intializeRandomPieceFromPreset("default", 200, getPieces("bwv99.6"))
print(piece)
generateFromModel("default", piece, savePath = "generated5.mid", iterations = 20, sample = False)

