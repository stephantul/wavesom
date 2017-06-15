import numpy as np

from collections import defaultdict, Counter
from string import ascii_lowercase
from .preprocessing.onc.ipapy_onc import ipapy_onc
from .preprocessing.orthographizer.mcclelland import mcclelland_orthography
from .preprocessing.sampler import random_sample


def setup(words, max_len, wordlist, num_to_sample=10000):
    """
    Set up a set of words through sampling.

    Sampling is done based on frequency of the word.

    :param words: a list of words, where each is represented as a tuple
    containing orthography, frequency, the language, and the
    syllable structure of that word.
    :max_len: The maximum length of a word in number of letters.
    :wordlist: A wordlist to sample from.
    :num_to_sample: The number of words to sample in total.
    """
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

    _, _, _, syll = zip(*words)

    i.fit(syll)

    for ortho, freq, lang, syll in words:

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
            print("Skipped {0}".format(ortho, str(syll)))
            print(e)
            continue

        listo.append(" ".join((ortho, str(syll))))
        X.append(np.hstack([orth, x_phon]))

        w2id[ortho].append(idx)
        word_dict[ortho].append(str(syll))

        frequencies.append(int(freq))
        idx += 1

    X_unique = np.array(X)
    frequencies = np.log(np.array(frequencies) + 1)

    sample = np.array(random_sample(listo,
                                    frequencies,
                                    num_to_sample,
                                    min_count=10))
    unique_words = list(set(sample))
    counts = Counter(sample)

    X = np.array([X_unique[listo.index(item)] for item in sample])
    X_unique = np.array([X_unique[listo.index(x)] for x in unique_words])

    return X, X_unique, unique_words, counts
