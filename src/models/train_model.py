from build_model import buildModel
from keras.callbacks import EarlyStopping
from create_datagen import *
from keras.models import load_model
import argparse
import json
from create_datagen import DATAGEN_PATH
MODEL_PATH = "models/"



def trainModel(trainDatagen, testDatagen, savePath, fitConfig, buildConfig, compileConfig = {"optimizer": "adam", "loss": "categorical_crossentropy", "metrics":["accuracy"]}):
    xTest = testDatagen[0][0]
    inputDimensions = [len(xInp[0]) for xInp in xTest]
    embedMaxDimensions = [len(cats) for cats in trainDatagen.sampler.oe.categories]
    softmaxDimension = len(trainDatagen.sampler.ohe.categories[0])
    
    buildConfig["inputDimensions"] = inputDimensions
    buildConfig["embeddingMaxDimensions"] = embedMaxDimensions
    buildConfig["softmaxDimension"] = softmaxDimension

    model = buildModel(**buildConfig)
    model.compile(**compileConfig)
    history = model.fit(trainDatagen, validation_data = testDatagen, **fitConfig)
    
    folder = MODEL_PATH+savePath
    model.save(folder+"/model.h5")
    trainDatagen.save(folder+"/train_datagen.pickle")
    testDatagen.save(folder+"/test_datagen.pickle")





if (__name__ == "__main__"):
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("train datagen", help = f"CSV path of train data relative to data/")
    my_parser.add_argument("test datagen", help = f"CSV path of test data relative to data/")
    my_parser.add_argument("model save path", help = f"Where the model will be saved relative to models/")
    my_parser.add_argument("embedding dimension", help = f"Output dimension of embeddings")
    my_parser.add_argument("lstm1 dimension", help = f"Right and left LSTM1 output dimension")
    my_parser.add_argument("middle dimension", help = f"Middle fully connected layer output dimension")


    my_parser.add_argument("--lstm1rd", help = f"Right and left LSTM1 recurrent dropout", default=0)
    my_parser.add_argument("--lstm1d", help = f"Right and left LSTM1 dropout", default=0)

    my_parser.add_argument("--lstm2dim", help = f"Right and left LSTM2 output dimension", default=None)
    my_parser.add_argument("--lstm2rd", help = f"Right and left LSTM2 recurrent dropout", default=0)
    my_parser.add_argument("--lstm2d", help = f"Right and left LSTM2 dropout", default=0)
    my_parser.add_argument("--middled", help = f"Middle dropout", default=None)
    
    my_parser.add_argument("--epochs", help = f"Number of epochs", default=500)

    args = my_parser.parse_args()

    args = vars(args)

    datagenTest = Datagen.load("data/"+args["test datagen"])
    datagenTrain = Datagen.load("data/"+args["train datagen"])


    buildConfig = {}
    buildConfig["LSTM1Dimension"] = int(args["lstm1 dimension"])
    buildConfig["LSTM2Dimension"] = None if args["lstm2dim"] == None else int(args["lstm2dim"])
    buildConfig["middleDimension"] = int(args["middle dimension"])
    buildConfig["LSTM1Dropout"] = float(args["lstm1d"])
    buildConfig["LSTM2Dropout"] = float(args["lstm2d"])
    buildConfig["LSTM1RecurrentDropout"] = float(args["lstm1rd"])
    buildConfig["LSTM2RecurrentDropout"] = float(args["lstm2rd"])
    buildConfig["middleDropout"] = None if args["middled"] == None else float(args["middled"])
    buildConfig["embeddingOutputDimension"] = int(args["embedding dimension"])

    f = open(MODEL_PATH + args["model save path"] + "/buildConfig.json", "w")
    json.dump(buildConfig, f)
    f.close()

    trainModel(datagenTrain, datagenTest, args["model save path"], {"epochs": int(args["epochs"]), "steps_per_epoch": 200, "validation_steps": 20, "callbacks":[EarlyStopping(patience=10, restore_best_weights=True)]}, buildConfig)


    




