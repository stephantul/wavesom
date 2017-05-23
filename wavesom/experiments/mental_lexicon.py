import json
import time
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
from wavesom.experiments.preprocessing.sivi.ipapy_features import ipapy_sivi

from wavesom.experiments.preprocessing.bow_encoder import BowEncoder
from wavesom.experiments.preprocessing.sampler import random_sample
from wavesom.wavesom import Wavesom


def check(sample_set, s, o):

    res = []

    for w in sample_set:
        w, _ = w.split()
        d = np.argmin(s.propagate(o.transform(w).ravel()[np.newaxis, :])[0])
        if d in pas[w]:
            res.append(True)
        else:
            res.append(False)

    return res


def transfortho(x, o, length):

    zero = np.zeros((length,))
    vec = o.transform(x).ravel()
    zero[:len(vec)] += vec

    return zero

if __name__ == "__main__":

    english_dict = json.load(open("data/english.json"))

    max_len = 6

    sivi = ipapy_sivi(4)
    o = BowEncoder()
    orth_vec_len = max_len * o.datalen
    np.random.seed(44)

    listo = []
    X = []

    idx = 0

    w2id = defaultdict(list)

    frequencies = []

    for ortho, freq, phono in english_dict:

        if len(set(ortho) - set(ascii_lowercase)) > 0:
            continue

        if len(ortho) > max_len:
            continue

        if freq < 100:
            continue

        orth = np.zeros((orth_vec_len,))
        x_orth = o.transform(ortho).ravel()

        orth[:x_orth.shape[0]] = x_orth

        try:
            x_phon = sivi.vectorize_single(phono, left=False)
        except ValueError as e:
            continue

        listo.append(" ".join((ortho, phono)))
        X.append(np.hstack([orth, x_phon]))

        w2id[ortho].append(idx)
        frequencies.append(freq)
        idx += 1

    X_orig = np.array(X)
    frequencies = np.log(np.array(frequencies) + 1)

    sample = np.array(random_sample(listo[:1000], frequencies[:1000], 100000))

    s__ = list(set(sample))

    X = np.array([X_orig[listo.index(item)] for item in sample])
    X_orig = np.array([X_orig[listo.index(x)] for x in s__])

    start = time.time()

    s = Wavesom((20, 20), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
    s.train(X, 100, total_updates=10000, batch_size=100, show_progressbar=True, stop_nb_updates=0.5)

    # TODO: increase map size, play with hyperparams
    # TODO: think of feedback mechanism

    print("Took {0} seconds".format(time.time() - start))

    sap = defaultdict(list)
    pas = defaultdict(list)

    sample_set = set(sample)

    for k, v in zip(s__, s.predict(X_orig)):

        if k not in sample_set:
            continue

        orth, pron = k.split()

        sap[v].append(k)
        pas[orth].append(v)

    print(w2id['wind'])
    # res = check(sample_set, s, o)
    # print(len([x for x in res if x]) / len(res))