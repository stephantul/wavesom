import numpy as np
import re


class Onc(object):

    def __init__(self, vowels, consonants, max_syll_length=3):
        """

        :param phonemes:
        :param max_length:
        """

        self.vowels = vowels
        self.consonants = consonants

        self.vowel_length = len(next(self.vowels.values().__iter__()))
        self.consonant_length = len(next(self.consonants.values().__iter__()))

        self.num_syls = 0
        self.o = 0
        self.n = 0
        self.c = 0
        self.veclen = 0
        self.syl_len = 0
        self.max_syll_length = max_syll_length

        self.r = re.compile(r"V+")

        # Assume all phonemes can occur in all positions.

    def fit(self, corpus):
        """
        Calculate maximum ONC grid

        :param corpus:
        :return:
        """

        num_syls = 0
        o = n = c = 0

        for syll in corpus:

            if len(syll) > self.max_syll_length:
                continue

            num_syls = max(num_syls, len(syll))

            for cvc in syll:

                cvc = [x for x in cvc if x in self.vowels or x in self.consonants]
                cvc = "".join(["C" if x in self.consonants else "V" for x in cvc])
                c_l = len(cvc)
                try:
                    m = next(self.r.finditer(cvc))
                    o = max(m.start(), o)
                    n = max(len(m.group()), n)
                    c = max(c_l - m.end(), c)
                except StopIteration:
                    o = max(c_l, o)

        self.num_syls = num_syls
        self.o = o
        self.n = n
        self.c = c

        self.syl_len = (o * self.consonant_length) + (n * self.vowel_length) + (c * self.consonant_length)
        self.veclen = self.syl_len * num_syls

    def vectorize_single(self, word):

        return self.vectorize([word])[0]

    def vectorize(self, words):

        o_vec = self.o * self.consonant_length
        n_vec = (self.n * self.vowel_length) + o_vec

        total = []

        for syll in words:

            if len(syll) > self.max_syll_length:
                raise ValueError("{0} is too long".format(syll))

            X = []

            for idx in range(self.num_syls):

                syll_vec = np.zeros(self.syl_len)

                try:
                    s = syll[idx]
                except IndexError:
                    X.append(syll_vec)
                    continue
                cvc = "".join(["C" if x in self.consonants else "V" for x in s])
                try:
                    m = next(self.r.finditer(cvc))
                    o = m.start()
                    n = len(m.group()) + o

                    for idx, x in enumerate(s[:o]):
                        syll_vec[idx * self.consonant_length: (idx+1)*self.consonant_length] = self.consonants[x]

                    for idx, x in enumerate(s[o:n]):
                        syll_vec[(idx * self.vowel_length) + o_vec: ((idx+1)*self.vowel_length) + o_vec] = self.vowels[x]

                    for idx, x in enumerate(s[n:]):
                        syll_vec[(idx * self.consonant_length) + n_vec: ((idx+1) * self.consonant_length) + n_vec] = self.consonants[x]

                except StopIteration:

                    for idx, x in enumerate(s[:len(cvc)]):
                        syll_vec[idx * self.consonant_length: (idx+1)*self.consonant_length] = self.consonants[x]

                X.append(syll_vec)

            total.append(np.array(X).ravel())

        return np.array(total)
