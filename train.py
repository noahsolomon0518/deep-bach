from os import name
from argparse import ArgumentParser
from keras.callbacks import CSVLogger, EarlyStopping
from pathlib import Path
from src.dataset.dataset import BachChorale, Datagen, getTimeSignature
from src.dataset.meta_manager import *
from src.model import build_model
from src.generate import generate
from tqdm import tqdm
import random
import os
from keras.models import load_model

def get_all_chorales():
    paths = music21.corpus.getComposer('bach')
    parsed = []
    for path in tqdm(paths):
        piece = music21.corpus.parse(path)
        if(set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in piece.parts]) and getTimeSignature(piece)==["4/4"]):
            parsed.append(piece)
    random.shuffle(parsed)
    return parsed


def get_test_chorales():
    paths = music21.corpus.getComposer('bach')
    parsed = []
    for path in tqdm(paths[:10]):
        piece = music21.corpus.parse(path)
        if(set(['Soprano', 'Alto', 'Tenor', 'Bass'])==set([part.id for part in piece.parts]) and getTimeSignature(piece)==["4/4"]):
            parsed.append(piece)
    random.shuffle(parsed)
    return parsed

def getConfig(name):

    return dict(
        steps_per_epoch = 10000,
        validation_steps = 10000,
        callbacks = [
            CSVLogger("models/"+name+"/score_log.csv"),
            EarlyStopping(patience=10)
        ],
        epochs = 60
    )




"""
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
        
        Datagen.load("models/"+saveName+"/datagen.dg")

        f = open("models/"+saveName+"/config.txt", "w")
        f.write("transpositionRange: "+ str(_datagen.transpositionRange))
        f.close()
"""
        
model_config = {
    "bach_default": {
        "name": "bach_default",
        "chorales_func": get_all_chorales,
        "meta_managers":[BeatMeta(), FermataMeta(), KeyMeta()],
        "sequence_length": 16,
        "batch_size": 2000,
        "overwrite": False
    },
    "bach_test": {
        "name": "bach_test",
        "chorales_func": get_test_chorales,
        "meta_managers":[BeatMeta(), FermataMeta(), KeyMeta()],
        "sequence_length": 16,
        "batch_size": 1000,
        "overwrite": False
    }
}

"""Loads a model. If not exist trains a new one"""
def get_trained_model_and_dataset(model_name, overwrite = False):
    config = model_config[model_name]
    dataset = BachChorale(**config)
    dataset.create_dataset()
    if(os.path.exists("models/"+model_name+"/model.h5") and overwrite==False):
        return load_model("models/"+model_name+"/model.h5"), dataset

    model = build_model(dataset, 20)
    datagen = dataset.datagen
    model.fit(datagen, validation_data = datagen.test_dataset, epochs = 15)
    model.save("models/"+model_name+"/model.h5")
    return model, dataset


def test():
    dataset = BachChorale(**model_config["bach_test"])
    dataset.create_dataset()
    dataset.intialize_random_chorale(100)


if __name__ == "__main__":
    model, dataset = get_trained_model_and_dataset("bach_test")
    print(generate(model, dataset))
    


