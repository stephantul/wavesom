import numpy as np


class Orthographizer(object):
    """
    A vectorizer to convert characters to binary vectors.

    Orthographizer is meant to be used in models which require a
    measure of orthographic similarity based on visual appearance,
    i.e. the presence or absence of line segments. Such an approach
    is in line with work on word reading, most notably the Interactive
    Activation (IA) models by Mcclelland and Rumelhart.
    """

    def __init__(self, features):
        """
        A vectorizer to convert characters to binary vectors.

        An orthographic vectorizer which vectorizes according to the 14-segment
        display described in the original IA paper.
        It reserves 14 binary features for bars in various positions.
        """

        self.dlen = max([len(x) for x in features.values()])
        self.data = {k: self._convert_bin(v) for k, v in features.items()}

    def _convert_bin(self, coordinates):
        """
        Convert the numerical coordinates to a binary format.

        :param coordinates: A list of numerical coordinates.
        :return: A flat numpy array, representing the  converted form.
        """
        d = np.zeros((self.dlen,))
        for c in coordinates:
            d[c] += 1

        return d

    def _overlap(self, x):
        """
        Check the overlap between data and x and return the difference.

        :param x: The string to check
        :return: A set of characters occurring in the input that
        do not occur in the data
        """
        return set(x).difference(self.data.keys())

    def vectorize(self, sequence, check=True):
        """
        Vectorize a single word.

        Raises a ValueError if the word is too long.

        :param sequence: A string of characters
        :param check: whether to perform checking for illegal characters.
        :return: A numpy array, representing the
        concatenated letter representations.
        """
        if check:

            not_overlap = self._overlap(sequence)
            if not_overlap:
                raise ValueError("The sequence contained illegal characters: {0}".format(not_overlap))

        x = np.zeros((len(sequence), self.dlen,))
        for idx, c in enumerate(sequence):
            x[idx] += self.data[c]

        return x

    def transform(self, sequences, check=True):
        """
        Transform a list of words into a list of arrays.

        This function returns a list of arrays because not all
        arrays may have the same size.

        :param sequences: a list of strings
        :param check: whether to perform checking for illegal characters
        :return: A list containing arrays representing
        concatenated letter representations of all words in the sequence.
        """
        return [self.vectorize(s, check=check) for s in sequences]
