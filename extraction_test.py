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
        X[X == 0] = -1.

        s = Wavesom((10, 10), X.shape[1], 1.0, orth_len=a.shape[1], phon_len=a.shape[1])
        s.fit(X, num_epochs=100, updates_epoch=1, batch_size=1)
        print("trained")
        start = time.time()

        X_ = X

        words = [str(x) for x in np.arange(len(X))]

        representations = s.converge(X_[:, :s.orth_len], batch_size=250, tol=0.01)
        r_ = np.copy(representations)
        representations = (s.weights[None, :, :] * representations[:, :, None]).mean(1)

        representations[representations <= .0] = -1.
        representations[representations > .0] = 1.0

        print(representations.shape, X.shape)
        score = len(X[np.all(X == representations, axis=1)]) / len(X)
        scores[cardinality] = (score,)

        print("Took {} seconds".format(time.time() - start))
