import re
import numpy as np
import logging

from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.experiments.preprocessing.gecco import GeccoReader
from somplay.recursive import Recursive
from unidecode import unidecode

removenumbers = re.compile(r"\W")


if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)

    o = Orthographizer()

    g_nl = GeccoReader("data/L1ReadingData.csv")
    g_en = GeccoReader("data/L2ReadingData.csv")

    nl_corpus = " ".join(g_nl.corpus[0])
    #en_corpus = " ".join(g_en.corpus[0])
    #corpus = " ".join([nl_corpus, en_corpus])

    #del nl_corpus
    #del en_corpus

    corpus = removenumbers.sub(" ", nl_corpus)
    corpus = unidecode(corpus)
    X = o.transform(corpus)

    r = Recursive((20, 20), 17, 0.3, alpha=2.0, beta=1.0)
    r.train(X, num_epochs=10, total_updates=1000, show_progressbar=True)

    whole = lambda x: r.predict(o.transform(x))