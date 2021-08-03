from os import name
from src.model import trainPresetModel
from src.datagen_config import config
from argparse import ArgumentParser
from keras.callbacks import CSVLogger, EarlyStopping
from pathlib import Path

CONFIG = config

def getConfig(name):

    return dict(
        steps_per_epoch = 50,
        validation_steps = 25,
        callbacks = [
            CSVLogger("models/"+name+"/score_log.csv"),
            EarlyStopping(patience=10)
        ],
        epochs = 500
    )


def getTestConfig(name):

    return dict(
        steps_per_epoch = 5,
        validation_steps = 5,
        callbacks = [
            CSVLogger("models/"+name+"/score_log.csv"),
            EarlyStopping(patience=10)
        ],
        epochs = 5
    )




def main():
    parser = ArgumentParser()
    parser.add_argument("--presets", action = "extend", nargs="+", type=str)
    parser.add_argument("--preset_options", action = "store_true", default = False)
    parser.add_argument("--test", action = "store_true", default = False)
    args = vars(parser.parse_args())
    if(args["preset_options"]==True):
        print("Preset Options:")
        print()
        for key,value in CONFIG.items():
            print("   ", key)
            for key2, value2 in value.items():
                    print("      ", str(key2) + ":" + str(value2))
            print()
        return

    modelNames = args["presets"]
    saveNames = ["test_"+modelName for modelName in modelNames] if args["test"] else modelNames
    for modelName,saveName in zip(modelNames,saveNames):
        Path("models/"+saveName).mkdir(parents=True, exist_ok=True)
        print("TRAINING MODEL:", modelName)
        fitConfig = getTestConfig(saveName) if args["test"] else getConfig(saveName)
        model, _datagen = trainPresetModel(modelName, fitConfig=fitConfig)
        model.save("models/"+saveName+"/model.h5")
        _datagen.save("models/"+saveName+"/datagen.dg")






if __name__ == "__main__":
    main()
    


