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

    dicto = [k for k in dicto if k[2] != 'dut']

    max_len = 7

    if args.file is None:
        raise ValueError("No file is given")

    X, X_orig, s_ = setup(dicto, max_len, [], 100000)

    orth_vec_len = 14 * max_len

    start = time.time()

    if args.gpu is not None:

        with cp.cuda.Device(args.gpu):
            X = cp.asarray(X, cp.float32)
            s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
            s.fit(X, args.epochs, total_updates=100, batch_size=250, show_progressbar=True, stop_nb_updates=0.5)

    else:

        s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
        s.fit(X, args.epochs, total_updates=100, batch_size=250, show_progressbar=True, stop_nb_updates=0.5)

    s.save(args.file)
