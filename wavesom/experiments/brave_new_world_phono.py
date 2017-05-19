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

from string import ascii_lowercase


removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)

    brv = " ".join(open("data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)

    r = re.compile("\n")
    brv = r.sub(" ", brv)
    brv = " ".join(brv.split())[158:]

    english = json.load(open("data/english.json"))

    new_brv = []
    phon_brv = []

    for word in brv.split():
        try:
            phon_brv.append(english[word])
            new_brv.append(word)
        except KeyError:
            print(word)
            continue

    brv = " ".join(new_brv)
    phon_brv = " ".join(phon_brv)

    mask = reset_context_symbol(brv, [" "])

    print("{0}".format(len(brv)))

    s = ipapy_sivi(3)

    o = Orthographizer()
    X_orth = np.array(o.fit_transform(brv))
    X_phon = np.array([s.phonemes[x] if x != " " else np.zeros((3,)) for x in phon_brv])
    start = time.time()
    r = Recursive((20, 20), X_phon.shape[1], learning_rate=0.1, alpha=3, beta=1, nbfunc=expo)
    r.train(X_phon, num_epochs=10, total_updates=1000, stop_nb_updates=0.5, show_progressbar=True)
    print("Took {0} seconds".format(time.time() - start))
    rec, _ = r.receptive_field(X_phon, phon_brv, 10)
