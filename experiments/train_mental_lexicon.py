import json
import time
import argparse

from wavesom.wavesom import Wavesom
from experiments.setup import setup
import cupy as cp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="The gpu to use", default=None)
    parser.add_argument("--file", type=str, help="The path to save the trained model to")
    parser.add_argument("--dim", type=int, help="The map dimensionality", default=30)
    parser.add_argument("--epochs", type=int, help="The number of epochs to train", default=100)
    args = parser.parse_args()

    dicto = json.load(open("data/syllable_lexicon.json"))

    max_len = 7

    wordlist = {'kind', 'mind', 'bind',
                'room', 'wolf', 'way',
                'wee', 'eten', 'eating',
                'wind', 'lead', 'speed',
                'meat', 'meet', 'meten',
                'spring', 'winter', 'spel',
                'speel', 'spill', 'spoon',
                'moon', 'boon', 'doen',
                'biet', 'beat', 'spijt',
                'tijd', 'mijt', 'klein',
                'fijn', 'rein', 'reign',
                'feign', 'fine', 'mine',
                'dine', 'wine', 'wijn',
                'weinig', 'speel', 'meel',
                'keel', 'veel', 'kiel',
                'wiel', 'wheel', 'feel',
                'deal', 'spool', 'spoel',
                'trol', 'troll', 'wolf',
                'rond', 'mond', 'mound',
                'hound', 'found', 'find',
                'lined', 'spine', 'mine',
                'cake', 'keek', 'leem',
                'dessert', 'desert', 'model',
                'curve', 'dies', 'nies', 'vies',
                'boot', 'potent', 'sate',
                'tree', 'rapport', 'lied',
                'bied', 'mond', 'kei',
                'steen', 'meen', 'geen',
                'keek', 'leek', 'gek',
                'creek', 'ziek', 'piek'}

    X, X_orig, s_, c = setup(dicto, max_len, wordlist)
    orth_vec_len = 14 * max_len

    start = time.time()

    if args.gpu:

        with cp.cuda.Device(args.gpu):
            X = cp.asarray(X, cp.float32)
            s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
            s.fit(X, args.epochs, total_updates=100, batch_size=250, show_progressbar=True, stop_nb_updates=0.5)

    else:

        s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
        s.fit(X, args.epochs, total_updates=100, batch_size=250, show_progressbar=True, stop_nb_updates=0.5)

    s.save(args.file)
