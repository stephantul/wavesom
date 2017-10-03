import numpy as np
import time

from wavesom import Wavesom
from itertools import product
from experiments.evaluation import evaluate_symbolic, evaluate_distributed


if __name__ == "__main__":

    scores = {}

    for cardinality in range(6, 14, 1):

        print(cardinality)

        a = np.array(list(product((0, 1), repeat=cardinality)))
        X = np.concatenate([a, a], 1)
        X = np.sign(X)

        s = Wavesom((10, 10), X.shape[1], 1.0)
        s.fit(X, num_epochs=100, updates_epoch=1, batch_size=1)
        print("trained")
        start = time.time()

        words = [str(x) for x in np.arange(len(X))]

        representations = s.converge(X[:, :X.shape[1]//2], batch_size=250, tol=0.01)
        representations = s.statify(representations)

        print(representations.shape, X.shape)
        score = len(X[np.all(X == representations, axis=1)]) / len(X)
        scores[cardinality] = (score,)

        print("Took {} seconds".format(time.time() - start))
