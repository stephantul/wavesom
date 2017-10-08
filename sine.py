"""Zoot suit."""
import numpy as np

from sklearn.cluster import KMeans
from wavesom import Wavesom

def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)[:, None]
    return dist

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    X = np.squeeze(np.stack([np.sin(np.arange(1000)/10)[:, None], np.sin(np.arange(1000)/5)[:, None]])).T
    X -= X.min(0)[None, :]
    X = X / 2.0

    k = KMeans(25)
    k.fit(X)

    scores = {}

    s = Wavesom((k.cluster_centers_.shape[0], 1), X.shape[1], 1.0, dampening=0.9)
    s.weights = k.cluster_centers_
    # zeppos = s.statify(s.converge(X))

    dist = s.predict_distance_part(X, 0)
    x = softmax(dist)
    x = s.statify(x)
    x -= x.min(0)[None, :]
    x_rep = x * (X.max(0) / x.max(0))

    res, idx = s.converge(X[:, :1], batch_size=32, max_iter=1000)
    rep = s.statify(res)
    rep -= rep.min(0)[None, :]
    rep = rep * (X.max(0) / rep.max(0))

    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(rep[:, 0], rep[:, 1], color='red')
    plt.scatter(s.weights[:, 0], s.weights[:, 1])
    plt.scatter(x_rep[:, 0], x_rep[:, 1])
    # plt.scatter(zeppos[:, 0], zeppos[:, 1])
