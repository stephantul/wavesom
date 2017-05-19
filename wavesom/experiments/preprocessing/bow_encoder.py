import numpy as np
from string import ascii_lowercase


class BowEncoder(object):

    def __init__(self, features=ascii_lowercase):

        self.data = np.zeros((len(features), len(features)))
        self.data[np.diag_indices(len(features), 2)] = 1

        self.data = {feat: self.data[idx] for idx, feat in enumerate(features)}
        self.datalen = len(list(self.data.values())[0])

    def transform(self, sequence):
        """
        Vectorize a single word.

        Raises a ValueError if the word is too long.

        :param sequence: A string of characters
        :param check: whether to perform checking for illegal characters.
        :return: A numpy array, representing the concatenated letter representations.
        """

        x = np.zeros((len(sequence), len(self.data),))
        for idx, c in enumerate(sequence):
            x[idx] += self.data[c]

        return x
