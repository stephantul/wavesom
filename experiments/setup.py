import numpy as np

from string import ascii_uppercase
from wordkit.feature_extractors import binary_character_features, phoneme_features
from wordkit.transformers import Orthographizer, Onc
from wordkit import WordKit
from wordkit.features.orthographical import mcclelland
from wordkit.features.phonological import patpho_bin
from .preprocessing.sampler import random_sample


def setup(words, max_len, wordlist, num_to_sample=10000, min_freq=10):
    """
    Set up a set of words through sampling.

    Sampling is done based on frequency of the word.

    :param words: a list of words, where each is represented as a tuple
    containing orthography, frequency, the language, and the
    syllable structure of that word.
    :max_len: The maximum length of a word in number of letters.
    :wordlist: A wordlist to sample from.
    :num_to_sample: The number of words to sample in total.
    :min_freq: The minimum number of frequency a certain word can occur with.
    :return: A matrix of words, Every unique word as a vector,
    Every unique word, and the word counts.
    """
    wordlist = set(wordlist)

    vowel_features, consonant_features = phoneme_features(patpho_bin)
    character_features = binary_character_features(mcclelland)

    orth = Orthographizer(character_features)
    onc = Onc(vowel_features, consonant_features)

    np.random.seed(44)

    unique_words = []
    frequencies = []
    already_added = set()

    for o, freq, _, syll in words:

        o = o.upper()
        if min_freq > freq:
            continue
        if len(o) > max_len:
            continue
        if wordlist and o.lower() not in wordlist:
            continue
        if set(o) - set(ascii_uppercase):
            continue
        form = " ".join((o, "-".join(syll)))
        if form in already_added:
            continue
        already_added.add(form)
        unique_words.append((o, syll))
        frequencies.append(freq)

    w = WordKit(character_features,
                vowel_features,
                consonant_features,
                orth,
                onc)

    w.fit(unique_words)

    X_unique = np.hstack(w.transform(unique_words))
    X = np.array(random_sample(X_unique, frequencies, 10000))

    return X, X_unique, unique_words
