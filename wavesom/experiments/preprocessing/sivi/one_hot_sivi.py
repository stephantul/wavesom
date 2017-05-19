import numpy as np

from ipapy.ipastring import IPAString
from .sivi import Sivi


def one_hot_sivi(length, phoneset="ɡæsɪrbʌɒɑtyðəvepʒuhʃoxdɛiwθjlɔʊmnaŋɜkz"):
    """
    Return a SIVI instance with one-hot features.

    :param length: The length of the grid in the sivi instance.
    :param phoneset: The phoneme set to use.
    :return: A sivi instance.
    """

    vowels, consonants = featurize_phoneset(phoneset)
    return Sivi(vowels, consonants, max_length=length)


def featurize_phoneset(phoneset):
    """
    Featurize the phonetic features into a given binary representation.

    :param phoneset: A list of phonemes
    :return: A dictionary of featurized phonemes and a list of vowels.
    """

    phonemes = IPAString(unicode_string="".join(phoneset), single_char_parsing=True)
    phonemes = [p for p in phonemes if not p.is_suprasegmental and not p.is_diacritic]

    num_vowels = len([p for p in phonemes if p.is_vowel])
    num_consonants = len([p for p in phonemes if not p.is_vowel])

    v = np.zeros((num_vowels, num_vowels))
    v[np.diag_indices(num_vowels, 2)] = 1
    c = np.zeros((num_consonants, num_consonants))
    c[np.diag_indices(num_consonants, 2)] = 1

    vowels = {str(p): v[idx] for idx, p in enumerate([p for p in phonemes if p.is_vowel])}
    consonants = {str(p): c[idx] for idx, p in enumerate([p for p in phonemes if not p.is_vowel])}

    # Finalize into dict
    phonemes = vowels.copy()
    phonemes.update(consonants)

    return vowels, consonants
