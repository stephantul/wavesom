import numpy as np

from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAVowel
from .onc import Onc

D1 = {"vowel": 0.1,
      "voiced": 0.75,
      "voiceless": 1.0}

D2 = {"front": 0.1,
      "central": 0.175,
      "back": 0.25,
      "bilabial": 0.45,
      "labio-dental": 0.528,
      "dental": 0.606,
      "alveolar": 0.684,
      "palato-alveolar": 0.762,
      "palatal": 0.841,
      "velar": 0.921,
      "glottal": 1.0}

D3 = {"open": 0.1,
      "open-mid": 0.185,
      "mid": 0.270,
      "close-mid": 0.35,
      "close": 0.444,
      "nasal": 0.644,
      "stop": 0.733,
      "fricative": 0.822,
      "approximant": 0.911,
      "lateral": 1.0}

D1_bin = {"voiced": [0],
          "voiceless": [1]}

D2_bin = {"front": [0, 1],
          "central": [1, 1],
          "back": [1, 0],
          "bilabial": [0, 0, 0],
          "labio-dental": [0, 0, 1],
          "dental": [0, 1, 0],
          "alveolar": [0, 1, 1],
          "palato-alveolar": [1, 0, 0],
          "palatal": [1, 0, 1],
          "velar": [1, 1, 0],
          "glottal": [1, 1, 1]}

D3_bin = {"open": [0, 1, 1],
          "open-mid": [0, 0, 1],
          "mid": [1, 0, 1],
          "close-mid": [1, 1, 0],
          "close": [1, 0, 0],
          "nasal": [0, 0, 0, 1],
          "stop": [0, 0, 1, 0],
          "fricative": [0, 1, 0, 0],
          "approximant": [0, 0, 1, 1],
          "lateral": [0, 1, 1, 0],
          "plosive": [1, 0, 0, 0],
          "trill": [1, 0, 0, 1]}

D4_bin = {"rounded": [0],
          "unrounded": [1]}


def ipapy_onc(length, phoneset="ɡæsɪrbʌɒɑtyðəvepʒuhʃoxdɛfiwθjlɔʊmnaŋɜkz", binary=True):
    """
    Return a SIVI instance with ipapy features

    :param length: The length of the grid in the sivi instance.
    :param phoneset: The phoneme set to use.
    :param binary: Whether to use binary or real-valued features.
    :return: A sivi instance.
    """

    vowels, consonants = featurize_phoneset(phoneset, binary)
    return Onc(vowels, consonants)


def featurize_phoneset(phoneset, binary):
    """
    Featurize the phonetic features into a given binary representation.

    :param phoneset: A list of phonemes
    :return: A dictionary of featurized phonemes and a list of vowels.
    """

    phonemes = IPAString(unicode_string="".join(phoneset), single_char_parsing=True)
    phonemes = [p for p in phonemes if not p.is_suprasegmental and not p.is_diacritic]

    # Convert to feature-based representation.
    convert = convert_binary if binary else convert_real
    vowels = {str(p): convert(p) for p in [p for p in phonemes if p.is_vowel]}
    consonants = {str(p): convert(p) for p in [p for p in phonemes if not p.is_vowel]}

    return vowels, consonants


def convert_binary(phoneme):
    """
    Converts a single phoneme to a vector of features.

    :param phoneme: A phoneme as ipapy IPAChar.
    :return: A dictionary of features.
    """


    vec = []

    if type(phoneme) is IPAVowel:

        backness = phoneme.backness
        if backness == "near-back":
            backness = "back"
        elif backness == "near-front":
            backness = "front"

        height = phoneme.height

        if height == "near-open":
            height = "open"
        elif height == "near-close":
            height = "close"

        vec.extend(D2_bin[backness])
        vec.extend(D3_bin[height])
        vec.extend(D4_bin[phoneme.roundness])

    else:

        place = phoneme.place

        if place == "labio-velar":
            place = "velar"

        manner = phoneme.manner

        if manner in ["sibilant-fricative", "non-sibilant-fricative"]:
            manner = "fricative"
        elif manner == "lateral-approximant":
            manner = "approximant"

        vec.extend(D1_bin[phoneme.voicing])
        vec.extend(D2_bin[place])
        vec.extend(D3_bin[manner])

    return np.array(vec)


def convert_real(phoneme):
    """
    Converts a single phoneme to a vector of features.

    :param phoneme: A phoneme as ipapy IPAChar.
    :return: A dictionary of features.
    """

    vec = []

    if type(phoneme) is IPAVowel:

        backness = phoneme.backness
        if backness == "near-back":
            backness = "back"
        elif backness == "near-front":
            backness = "front"

        height = phoneme.height

        if height == "near-open":
            height = "open"
        elif height == "near-close":
            height = "close"

        vec.append(D1["vowel"])
        vec.append(D2[backness])
        vec.append(D3[height])

    else:

        place = phoneme.place

        if place == "labio-velar":
            place = "velar"

        manner = phoneme.manner

        if manner in ["sibilant-fricative", "non-sibilant-fricative"]:
            manner = "fricative"
        elif manner == "plosive":
            manner = "nasal"
        elif manner == "trill":
            manner = "nasal"
        elif manner == "lateral-approximant":
            manner = "approximant"

        vec.append(D1[phoneme.voicing])
        vec.append(D2[place])
        vec.append(D3[manner])

    return np.array(vec)

if __name__ == "__main__":

    import json

    p = json.load(open("syllable_lexicon.json"))
    o = ipapy_onc(3)
    o.fit(p, 3)
    z = o.vectorize(p)
    print("done")