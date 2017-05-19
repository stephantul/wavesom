import numpy as np
import json

from somber.batch.som import Som as Batch_Som
from somber.utils import expo, linear
from collections import defaultdict


def norm(vector):

    return np.square(vector / np.linalg.norm(vector))


class Wavesom(Batch_Som):

    def __init__(self,
                 map_dim,
                 dim,
                 learning_rate,
                 orth_len,
                 phon_len,
                 lrfunc=expo,
                 nbfunc=expo,
                 sigma=None,
                 min_max=np.argmin):

        super().__init__(map_dim, dim, learning_rate, lrfunc, nbfunc, sigma, min_max)

        self.orth_len = orth_len
        self.phon_len = phon_len

    @classmethod
    def load(cls, path):
        """
        Loads a Lexisom

        :param path: The path to the JSON file where the lexisom is stored
        :return: A lexisom.
        """
        data = json.load(open(path))

        weights = data['weights']
        weights = np.array(weights, dtype=np.float32)
        datadim = weights.shape[1]

        dimensions = data['dimensions']
        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        lr = data['lr']
        sigma = data['sigma']
        orth_len = data['orth_len']
        phon_len = data['orth_len']

        s = cls(dimensions, datadim, lr, orth_len, phon_len, lrfunc=lrfunc, nbfunc=nbfunc, sigma=sigma)
        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """
        Save a lexisom

        :param path: The path to save the lexisom to.
        :return: None
        """

        dicto = {}
        dicto['weights'] = [[float(w) for w in x] for x in self.weights]
        dicto['dimensions'] = self.map_dimensions
        dicto['lrfunc'] = 'expo' if self.lrfunc == expo else 'linear'
        dicto['nbfunc'] = 'expo' if self.nbfunc == expo else 'linear'
        dicto['lr'] = self.learning_rate
        dicto['sigma'] = self.sigma
        dicto['orth_len'] = self.orth_len
        dicto['phon_len'] = self.phon_len

        json.dump(dicto, open(path, 'w'))

    def _predict_base_part(self, X, offset):
        """
        Computes the prediction for part of the weights, specified by an offset

        :param X: The input data
        :param offset: The offset which is applied to axis 1 of X
        :return: An matrix containing the distance of each sample to each weight.
        """

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        datadim = X.shape[-1]
        X = self._create_batches(X, batch_size=1)

        temp_weights = self.weights[:, offset:offset+datadim]

        distances = []

        for x in X:

            distance = self._euclidean(x, weights=temp_weights)
            distances.extend(distance)

        return np.array(distances)

    def predict_part(self, X, offset, orth_vec_len=0):
        """
        Predict BMUs based on part of the weights

        :param X:
        :param offset:
        :param orth_vec_len:
        :return:
        """

        if orth_vec_len:
            X = X[:, :orth_vec_len]

        dist = self._predict_base_part(X, offset)
        return self.min_max(dist, axis=1)

    def propagate(self, vector, max_depth=5, num=3, start_idx=0, end_idx=None, numbers=True):
        """
        Propagates information through the network weights, simulating attractors.

        :param vector: The vector to propagate
        :param max_depth: The max depth to which to propagate
        :param num: The number of weights to consider at each step
        :param start_idx: The start index for the data
        :param end_idx: The end index for the data
        :param numbers: Whether to return indices or the raw weights
        :return: An dictionary of weights for each timestep.
        """

        result = []
        mask = []

        self.rec_propagate(vector, result, mask, max_depth, num, start_idx, end_idx)

        dicto = defaultdict(list)

        for idx, res in zip(mask, result):
            if numbers:
                dicto[idx].append(np.argsort(-res)[:num])
            else:
                dicto[idx].append(res)

        return {k: np.array(v) for k, v in dicto.items()}

    def propagate_values(self, vector, max_depth=5, num=3, start_idx=0, end_idx=None):

        p = self.propagate(vector, max_depth, num, start_idx, end_idx, False)
        return {k: v.mean(axis=0) for k, v in p.items()}

    def get_value(self, vector, max_depth=5, num=3, start_idx=0, end_idx=None):

        s = self.propagate_values(vector, max_depth, num, start_idx, end_idx)
        return np.array(list(s.values())).sum(axis=0)

    def rec_propagate(self, x, result, mask, max_depth, num, startidx, end_idx, depth=0, strength=1.0):
        """
        A recursive propagation function.

        :param x:
        :param result: The
        :param mask:
        :param max_depth:
        :param num:
        :param startidx:
        :param end_idx:
        :param depth:
        :param strength:
        :return:
        """

        if depth == max_depth or strength < 0.1:
            return

        if end_idx is None:
            end_idx = self.data_dim

        res = np.zeros(self.weight_dim)

        if depth == 0:
            pred = self._predict_base_part(x[:self.orth_len], 0)[0]
            sort = pred.argsort()[:num]
            pred = 1.0 - (pred[sort] / pred[sort].max())

        else:
            pred = self._predict_base_part(x[startidx:end_idx], startidx)[0]
            sort = pred.argsort()[1:num+1]
            pred = 1.0 - (pred[sort] / pred[sort].max())

        res[sort] = pred * strength

        result.append(res)
        mask.append(depth)

        for x, p in zip(*[self.weights[sort], pred]):
            self.rec_propagate(x,
                               result,
                               mask,
                               max_depth,
                               num,
                               startidx,
                               end_idx,
                               depth=depth+1,
                               strength=(strength*p))
