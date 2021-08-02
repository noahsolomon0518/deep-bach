import argparse
from generation_utils import *
import pandas as pd
import time


pieces = [group for i, group in pd.read_csv("data/df/test_df.csv").groupby("pieceID")]

if __name__=="__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("piece number", help = f"Piece number ranging from 0 - 11")
    my_parser.add_argument("model name", help = f"Name of model's folder")
    my_parser.add_argument("iterations", help = f"How many times the piece is iterated over for generation")
    args = my_parser.parse_args()

    args = vars(args)


    pieceNum = int(args["piece number"])
    savePath = "models/"+args["model name"]+"/"+str(time.time())+".mid"
    iterations = int(args["iterations"])

    generateMusic(args["model name"], pieces[pieceNum], savePath, iterations)