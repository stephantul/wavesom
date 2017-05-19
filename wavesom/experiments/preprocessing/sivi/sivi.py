import numpy as np

from copy import copy


class Sivi(object):

    def __init__(self, vowels, consonants, max_length=3):
        """
        Python re-implementation of of PatPho, a system for converting sequences of phonemes to vector representations
        that capture phonological similarity of words.

        The system is described in:

            Li, P., & MacWhinney, B. (2002). PatPho: A phonological pattern generator for neural networks.
                Behavior Research Methods, Instruments, & Computers, 34(3), 408-415.

        The original C implementation can be found here (June 2015): http://www.personal.psu.edu/pul8/patpho_e.shtml

        :param vowels: a dictionary of vowels, the keys of which are
        characters, and the values of which are numpy arrays.
        :param consonants: a dictionary of consonsants, they keys of
        which are characters, and the values of which are numpy arrays.
        :param max_length: The length of the grid in "CO CO CO VO VO" clusters
        """

        self.vowel_length = len(list(vowels.values())[0])
        self.consonant_length = len(list(consonants.values())[0])

        if any([len(v) != self.vowel_length for v in vowels.values()]):
            raise ValueError("Not all vowel vectors have the same length")

        if any([len(v) != self.consonant_length for v in consonants.values()]):
            raise ValueError("Not all vowel vectors have the same length")

        self.max_length = max_length
        self.grid = self.init_grid()

        # create phoneme dictionary
        self.phonemes = copy(vowels)
        self.phonemes.update(consonants)

        # consonant dictionary
        self.consonants = consonants
        # vowel dictionary
        self.vowels = vowels

        # indexes
        self.index2consonant = {idx: c for idx, c in enumerate(self.consonants)}
        self.consonant2index = {v: k for k, v in self.index2consonant.items()}
        self.index2vowel = {idx: v for idx, v in enumerate(self.vowels)}
        self.vowel2index = {v: k for k, v in self.index2vowel.items()}
        self.phoneme2index = {p: idx for idx, p in enumerate(self.phonemes)}

        # The grid indexer contains the number of _Features_
        # the grid contains up to that point.
        self.grid_indexer = []

        # The phon indexer contains the number of phonemes
        # the grid contains up to that point.
        self.phon_indexer = []
        idx = 0
        idx_2 = 0
        for i in self.grid:
            if i == "CO":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.consonant_length
                idx_2 += len(self.consonant2index)
            elif i == "VO":
                self.grid_indexer.append(idx)
                self.phon_indexer.append(idx_2)
                idx += self.vowel_length
                idx_2 += len(self.vowel2index)

        # The total length of the grid in features
        self.grid_length = idx

    def _check(self, string):
        """
        Checks whether the given string you supply contains characters
        that were not specified in your phoneme set.

        :param string:
        :return:
        """

        return set(string) - set(self.phoneme2index.keys())

    def init_grid(self):
        """
        Initializes the syllabic grid

        :return: A list of 'CO' and 'VO' strings, representing
         whether that position in the grid represents a vowel
         or a consonant.
        """
        return ["CO", "CO", "CO", "VO", "VO"] * self.max_length + ["CO", "CO", "CO"]

    def vectorize_single(self, x, left=True):
        """
        Convert the phoneme sequence to a vector representation.

        :param x: A string representing phonemes
        :param left: whether to use left-justification
        """

        check = self._check(x)
        if check:
            raise ValueError("{0} contains invalid phonemes: {1}".format(x, " ".join(check)))

        grid = self._put_on_grid(x, left=left)

        # convert syllabic grid to vector
        phon_vector = np.zeros(self.grid_length)

        for idx, phon in grid.items():
            p = self.phonemes[phon]
            g_idx = self.grid_indexer[idx]
            phon_vector[g_idx: g_idx+len(p)] = p

        return np.array(phon_vector)

    def _put_on_grid(self, x, left=True):
        """
        Puts phonemes on a grid.

        :param x: A phonemic representation
        :param left: Whether to use left justification.
        :return: A dictionary where the keys are indices and the
        values are phonemes.
        """

        if not left:
            x = x[::-1]

        idx = 0
        indices = {}

        for p in x:

            grid = self.grid[idx:]

            if p in self.vowels:
                try:
                    nex = grid.index("VO")
                    if nex == 0:
                        nex = grid[1:].index("VO") + 1
                    idx += nex
                    indices[idx] = p

                except ValueError:
                    raise ValueError('Word is too long: {0}'.format(x if left else x[::-1]))

            elif p in self.consonants:
                try:
                    if idx == 0:
                        indices[0] = p
                        continue
                    nex = grid.index("CO")
                    if nex == 0:
                        nex = grid[1:].index("CO") + 1
                    idx += nex
                    indices[idx] = p
                except ValueError:
                    raise ValueError('Word is too long: {0}'.format(x if left else x[::-1]))
            else:
                raise ValueError("{0} not in lex".format(p))

        if not left:
            indices = {np.abs(k - (len(self.grid)-1)): v for k, v in indices.items()}

        return indices

    def grid_indices(self, x, left=True):
        """
        Gets the grid indices for a given phoneme input.

        Useful for determining whether two inputs overlap
        in their consonants and vowels (as opposed to their
        features)

        :param x: A phonemic representation
        :param left: Whether to use left-justification
        :return: A list of phoneme indices
        """

        grid = self._put_on_grid(x, left=left)
        return sorted([(self.consonant2index[v] if self.grid[k] == "CO" else self.vowel2index[v]) + self.phon_indexer[k] for k, v in grid.items()])

    def vectorize(self, X):
        """
        Vectorizes a set of words into a Numpy array.

        :param X: A list of strings, representing the phonological form of words.
        :return: A numpy array representing the phonological features.
        """
        results = []

        for x in X:
            results.append(self.vectorize_single(x))

        return np.array(results)
