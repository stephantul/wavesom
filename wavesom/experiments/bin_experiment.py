import numpy as np
import logging
import time
import cProfile

from somplay.batch.recursive import Recursive as BRecursive
from somplay.recursive import Recursive
from somplay.merging import Merging
from somplay.som import Som
from somplay.experiments.markov_chain import MarkovGenerator
from somplay.utils import static, MultiPlexer, linear, expo

if __name__ == "__main__":

    #logging.basicConfig(level=logging.INFO)

    # mgen = MarkovGenerator(np.array([[1, 0], [0, 1], [1, 1]]), np.array([[0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]), np.array([0.2, 0.4, 0.4]))
    mgen = MarkovGenerator(np.array([[0], [1]]), np.array([[0.3, 0.7], [0.4, 0.6]]), np.array([1.0, 0.0]))
    X = mgen.generate_sequences(1, 30000)[0]

    # X = np.random.binomial(1, 0.5, size=(1000000,))

    print("Generated data")
    print(X.shape)

    # X = np.random.binomial(1, 0.3, size=(1000000,))

    """start = time.time()
    r = Recursive((10, 10), 1, 0.1, alpha=1.1, beta=1.0, nbfunc=linear, lrfunc=linear)
    cProfile.run("r.train(X, 1, 1000, stop_nb_updates=0.66, batch_size=10, show_progressbar=True)")
    print("Took {0} seconds".format(time.time() - start))"""

    start = time.time()

    r = Recursive((10, 10), 1, 1.0, alpha=2, beta=1, nbfunc=linear, lrfunc=expo)
    r.train(X[:1000], 10, 1000, stop_nb_updates=0.66)

    m = Merging((10, 10), 1, 1.0, alpha=0.0, beta=0.5, nbfunc=linear, lrfunc=expo)
    m.train(X[:1000], 10, 1000, stop_nb_updates=0.66)
