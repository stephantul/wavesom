import numpy as np

from wavesom import Wavesom

if __name__ == "__main__":

    a = np.array([1, 0, 0, 1, 0, 0, 0])
    b = np.array([1, 0, 0, 0, 1, 0, 0])
    c = np.array([1, 0, 0, 0, 0, 1, 0])
    d = np.array([0, 1, 0, 0, 0, 1, 1])
    e = np.array([0, 0, 1, 0, 0, 0, 0])

    # X = np.concatenate([[a] * 1000, [b] * 1000, [c] * 1000, [d] * 2500, [e] * 2500])
    X_orig = np.vstack([a, b, c, d, e])

    s = Wavesom.load("saved_models/poc.json")
    s.state[:] = 1.0
    inv = s.invert_projection(X_orig, ['a', 'b', 'c', 'd', 'e'])

    spe = []
    spe.extend(s.activate(X_orig[0, :4]))
    spe.extend(s.activate(X_orig[1, :4]))
    spe.extend(s.activate(X_orig[3, :4]))

    spe = np.array(spe)
