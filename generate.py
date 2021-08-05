from src.generate import *






otherPiece = AIComposer.initPieceFromPreset("bwv10.7", "default")
otherPiece.noisify((32,64))
otherPiece.save("before.mid")
otherPiece.gibbsSample(iterations = 1000)
otherPiece.save("after.mid")
