import numpy as np

from collections import defaultdict, Counter
from string import ascii_lowercase
from .preprocessing.onc.ipapy_onc import ipapy_onc
from .preprocessing.orthographizer.mcclelland import mcclelland_orthography
from .preprocessing.sampler import random_sample


def setup(words, max_len, wordlist, num_to_sample=10000):

    wordlist = set(wordlist)

    i = ipapy_onc(3)
    o = mcclelland_orthography()
    orth_vec_len = max_len * len(o.data["A"])

    np.random.seed(44)

    listo = []
    X = []

    idx = 0

    w2id = defaultdict(list)

    frequencies = []
    word_dict = defaultdict(list)

    _, _, _, _, syll = zip(*words)

    i.fit(syll)

    for ortho, phono, freq, lang, syll in words:

        if ortho not in wordlist:
            continue

        if len(set(ortho) - set(ascii_lowercase)) > 0:
            continue

        if len(ortho) > max_len:
            print("{0} too long: {1}".format(ortho, len(ortho)))
            continue

        orth = np.zeros((orth_vec_len,))
        x_orth = np.ravel(o.vectorize(ortho.upper()))

        orth[:x_orth.shape[0]] = x_orth

        try:
            x_phon = i.vectorize_single(syll)
        except ValueError as e:
            print("Skipped {0}".format(ortho, phono))
            print(e)
            continue

        listo.append(" ".join((ortho, phono)))
        X.append(np.hstack([orth, x_phon]))

        w2id[ortho].append(idx)
        word_dict[ortho].append(phono)

        frequencies.append(int(freq))
        idx += 1

    X_orig = np.array(X)
    frequencies = np.log(np.array(frequencies) + 1)

    sample = np.array(random_sample(listo, frequencies, num_to_sample, min_count=10))
    s_ = list(set(sample))
    c_ = Counter(sample)

    X = np.array([X_orig[listo.index(item)] for item in sample])
    X_orig = np.array([X_orig[listo.index(x)] for x in s_])

    return X, X_orig, s_, c_