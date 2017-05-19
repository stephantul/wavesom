import json
import numpy as np
import time

from somplay.amlap.lexisom import Lexisom
from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.experiments.preprocessing.bow_encoder import BowEncoder
from somplay.experiments.preprocessing.sivi.ipapy_features import ipapy_sivi
from somplay.experiments.preprocessing.sampler import random_sample
# from somplay.experiments.preprocessing.sivi.binary_sivi import binary_sivi

from collections import Counter, defaultdict
from string import ascii_lowercase


def check(s__, s, o):

    res = []

    for w in s__:
        w, _ = w.split()
        d = np.argmin(s.propagate(o.transform(w).ravel()[np.newaxis, :])[0])
        res.append(d in pas[w])

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
    o = Orthographizer()
    orth_vec_len = max_len * o.datalen

    np.random.seed(44)

    listo = []
    X = []

    idx = 0

    w2id = defaultdict(list)

    frequencies = []

    '''words = {'wind', 'lead', 'grass',
             'gross', 'stem', 'room',
             'around', 'abound', 'sound',
             'land', 'wolf', 'pain',
             'wain', 'whine', 'dog',
             'cat', 'cream', 'walk',
             'dance', 'danced', 'walked',
             'bid', 'lid', 'bidding',
             'tread', 'thread', 'threat',
             'mind', 'shined', 'lined',
             'kind'}'''

    for ortho, freq, phono in english_dict:

        if len(set(ortho) - set(ascii_lowercase)) > 0:
            continue

        if len(ortho) > max_len:
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

        if idx == 100:
            break

    X_orig = np.array(X)
    frequencies = np.log(np.array(frequencies) + 1)

    sample = np.array(random_sample(listo[:200], frequencies[:200], 10000))
    s__ = list(set(sample))

    X = np.array([X_orig[listo.index(item)] for item in sample])
    X_orig = np.array([X_orig[listo.index(x)] for x in s__])

    start = time.time()

    s = Lexisom((10, 10), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
    s.train(X, 10, total_updates=1000, batch_size=100, show_progressbar=True, stop_nb_updates=0.5)

    # TODO: increase map size, play with hyperparams
    # TODO: think of feedback mechanism

    print("Took {0} seconds".format(time.time() - start))

    sap = defaultdict(list)
    pas = defaultdict(list)

    sample_set = set(sample)

    for k, v in zip(s__, s.predict(X_orig)):

        orth, pron = k.split()

        sap[v].append(k)
        pas[orth].append(v)

    print(w2id['wind'])
    # res = check(s__, s, o)
    # print(len([x for x in res if x]) / len(res))