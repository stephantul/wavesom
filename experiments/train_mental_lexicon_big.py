import json
import time
import argparse
import logging

from wavesom.wavesom import Wavesom
from wordkit.construct import WordkitPipeline
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, OrthographyTransformer
from wordkit.feature_extraction.features import fourteen, binary_features
from wordkit.feature_extraction import binary_character_features, phoneme_features
from wordkit.samplers import Sampler
from sklearn.pipeline import FeatureUnion

import cupy as cp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="The gpu to use", default=None)
    parser.add_argument("--file", type=str, help="The path to save the trained model to")
    parser.add_argument("--dim", type=int, help="The map dimensionality", default=30)
    parser.add_argument("--epochs", type=int, help="The number of epochs to train", default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    '''wordlist = {'kind', 'mind', 'bind',
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
                'creek', 'ziek', 'piek'}'''

    wordlist = json.load(open("data/shared_vocab.json"))
    wordlist = {x.lower() for x in wordlist if "-" not in x}

    o = OrthographyTransformer(binary_character_features(fourteen))
    vowels, consonants = phoneme_features(binary_features)
    p = ONCTransformer(vowels, consonants)

    transformers = ("t", FeatureUnion([("o", o), ("p", p)]))

    c_d = Celex("data/dpl.cd", language='dutch', merge_duplicates=True)
    c_e = Celex("data/epl.cd", merge_duplicates=True)

    corpora = ("corpora", FeatureUnion([("d", c_d), ("e", c_e)]))
    s = Sampler(num_samples=50000, include_all=True)

    construct = WordkitPipeline(stages=(corpora, transformers), sampler=s)
    (X, samples), (X_orig, words) = construct.fit_transform(wordlist, strict=False)

    orth_len = o.vec_len
    wordlist = list(wordlist)

    start = time.time()

    if args.gpu is not None:
        print("using GPU {}".format(args.gpu))
        with cp.cuda.Device(args.gpu):
            X = cp.asarray(X, cp.float32)
            s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_len, phon_len=X.shape[1] - orth_len)
            s.fit(X, args.epochs, total_updates=100, batch_size=1, init_pca=False, show_progressbar=True, stop_nb_updates=0.5)

    else:
        print("using CPU")
        s = Wavesom((args.dim, args.dim), X.shape[1], 1.0, orth_len=orth_len, phon_len=X.shape[1] - orth_len)
        s.fit(X, args.epochs, total_updates=100, batch_size=1, init_pca=False, show_progressbar=True, stop_nb_updates=1.)

    s.save(args.file)
