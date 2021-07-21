from music21 import corpus
import pickle
from tqdm import tqdm
import os



def main():
    paths = corpus.getComposer('bach')
    f = open("data/parsed_midis.pickle", "wb")
    parsed = []
    for path in tqdm(paths):
        parsed.append(corpus.parse(path))
    pickle.dump(parsed, f)

    f.close()
    print("Data dump complete.")


    
if __name__=="__main__":
    main()






