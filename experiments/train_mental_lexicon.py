import json
import time
import argparse
import logging

from wavesom.wavesom import Wavesom
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer
from wordkit.feature_extraction.features import fourteen, binary_features
from wordkit.feature_extraction import phoneme_features
from wordkit.samplers import Sampler
from sklearn.pipeline import FeatureUnion

import cupy as cp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="The gpu to use", default=None)
    parser.add_argument("--file", type=str, help="The path to save the trained model to")
    parser.add_argument("--dim", type=int, help="The map dimensionality", default=30)
    parser.add_argument("--epochs", type=int, help="The number of epochs to train", default=100)
    parser.add_argument("--batch", type=int, help="The batch size", default=1)
    parser.add_argument("--neighborhood", type=float, help="When to stop the neighborhood", default=1.0)
    parser.add_argument("--log", type=int, help="Whether to use log scaling", default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    wordlist = {'kind', 'mind', 'bind',
                'room', 'wolf', 'way',
                'wee', 'eten',
                'wind', 'lead', 'speed',
                'meat', 'meet', 'meten',
                'spring', 'winter', 'spel',
                'spill', 'spoon',
                'moon', 'boon', 'doen',
                'biet', 'beat', 'spijt',
                'tijd', 'mijt', 'klein',
                'fijn', 'rein', 'reign',
                'feign', 'fine', 'mine',
                'dine', 'wine', 'wijn',
                'weinig', 'meel',
                'keel', 'veel', 'kiel',
                'wiel', 'wheel', 'feel',
                'deal', 'spool', 'spoel',
                'trol', 'troll', 'wolf',
                'rond', 'mond', 'mound',
                'hound', 'found', 'find',
                'spine', 'mine',
                'cake', 'leem',
                'dessert', 'desert', 'model',
                'curve', 'dies', 'nies', 'vies',
                'boot', 'potent', 'sate',
                'tree', 'rapport', 'lied',
                'mond', 'kei',
                'steen', 'geen',
                'leek', 'gek',
                'creek', 'ziek', 'piek'}

    # wordlist = json.load(open("data/shared_vocab.json"))
    # wordlist = {x.lower() for x in wordlist if "-" not in x}

    o = LinearTransformer(fourteen)
    p = ONCTransformer(phoneme_features(binary_features))

    transformers = FeatureUnion([("o", o), ("p", p)])

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])
    s = Sampler(num_samples=10000, include_all=True, mode='log' if args.log else 'raw')

    from collections import Counter

    words_ = corpora.fit_transform(wordlist)
    X = transformers.fit_transform(words_)
    X, words = s.sample(X, words_)

    X[X == 0] = -1.

    print(Counter([x['orthography'] for x in words]))

    orth_len = o.vec_len
    wordlist = list(wordlist)

    start = time.time()

    if args.gpu is not None:
        print("using GPU {}".format(args.gpu))
        with cp.cuda.Device(args.gpu):
            X = cp.asarray(X, cp.float32)
            s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_len, phon_len=X.shape[1] - orth_len)
            s.fit(X, args.epochs, updates_epoch=1, batch_size=args.batch, show_progressbar=True, stop_nb_updates=args.neighborhood)

    else:
        print("using CPU")
        s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_len, phon_len=X.shape[1] - orth_len)
        s.fit(X, args.epochs, updates_epoch=1, batch_size=args.batch, show_progressbar=True, stop_nb_updates=args.neighborhood)

    s.save(args.file)
