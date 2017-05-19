import re
import numpy as np
import logging
import time
import json

from somplay.recursive import Recursive
from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.utils import linear, expo, reset_context_symbol
from somplay.experiments.preprocessing.corpus import read_carmel
from somplay.experiments.preprocessing.sivi.ipapy_features import ipapy_sivi
from collections import defaultdict

from string import ascii_lowercase


removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)

    brv = " ".join(open("data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)

    english_dict = json.load(open("data/english.json"))

    max_len = 7

    sivi = ipapy_sivi(3)
    o = Orthographizer()
    orth_vec_len = max_len * o.datalen

    np.random.seed(44)

    idx = 0

    X = []
    dicto = {}

    for ortho, freq, phono in english_dict:

        if ortho in dicto:
            continue

        if len(set(ortho) - set(ascii_lowercase)) > 0:
            continue

        if len(ortho) > max_len:
            print("{0} too long: {1}".format(ortho, len(ortho)))
            continue

        orth = np.zeros((orth_vec_len,))
        x_orth = o.transform(ortho).ravel()

        orth[:x_orth.shape[0]] = x_orth

        try:
            x_phon = sivi.vectorize_single(phono, left=False)
        except ValueError as e:
            print("Skipped {0}".format(ortho, phono))
            print(e)
            continue

        X.append(np.hstack([orth, x_phon]))

        dicto[ortho] = idx
        idx += 1

    X_orig = np.array(X)

    X = []
    sents = []

    for idx, sent in enumerate(brv.split("\n")):
        sent = sent.split()
        if len(set(sent) - set(dicto.keys())):
            continue
        if not sent:
            continue

        sents.append(sent)
        sent = [dicto[word] for word in sent]
        X.extend(X_orig[sent])

    X = np.array(X)

    print("{0}".format(len(brv)))

    start = time.time()
    r = Recursive((20, 20), X.shape[1], learning_rate=0.1, alpha=3, beta=1, nbfunc=expo)
    r.train(X, num_epochs=100, total_updates=1000, stop_nb_updates=0.5, show_progressbar=True)
    print("Took {0} seconds".format(time.time() - start))
