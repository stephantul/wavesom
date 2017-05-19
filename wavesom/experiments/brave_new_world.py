import re
import numpy as np
import logging
import time

from somplay.recursive import Recursive
from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.utils import linear, expo, reset_context_symbol

from string import ascii_lowercase


removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)

    brv = " ".join(open("data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)

    r = re.compile("\n")
    brv = r.sub(" ", brv)
    brv = " ".join(brv.split())[158:]

    mask = reset_context_symbol(brv, [" "])

    print("{0}".format(len(brv)))

    # words = brv.split()

    test = 'the cat sat on the mat mask master'

    # maxlen = 10
    # words = filter(lambda x: len(x) <= maxlen, words)

    o = Orthographizer()
    X = np.array(o.fit_transform(brv))

    print(X.shape)

    # X = np.hstack([X[:-2], X[1:-1], X[2:]])

    X_test = o.transform(test)

    settings = []

    for lr in np.linspace(0.01, 1.0, num=10):

        for alpha in np.linspace(0.5, 3.0, num=20):

            for beta in np.linspace(0.2, 2.0, num=20):

                if beta > alpha:
                    continue

                for nbfunc in [linear, expo]:

                    settings.append({'lr': lr, 'alpha': alpha, 'beta': beta, 'nbfunc': nbfunc})

    scores = {}

    for config in settings:

        lr = config['lr']
        alpha = config['alpha']
        beta = config['beta']
        nbfunc = config['nbfunc']

        start = time.time()
        r = Recursive((20, 20), X.shape[1], learning_rate=lr, alpha=alpha, beta=beta, nbfunc=nbfunc)
        r.train(X[:10000], num_epochs=10, total_updates=1000, stop_nb_updates=0.5, show_progressbar=False)
        print("Took {0} seconds".format(time.time() - start))
        rec, _ = r.receptive_field(X[:10000], brv[:10000], 10)
        s = sum([len(x) for x in rec.values()])
        scores[(lr, alpha, beta, str(nbfunc))] = (s / len(rec), s / r.map_dim)

        print(lr, alpha, beta, str(nbfunc), s / len(rec), s / r.map_dim)
