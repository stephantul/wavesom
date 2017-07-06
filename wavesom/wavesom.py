import numpy as np
import json

from somber import Som
from somber.utils import expo, linear, np_min
from collections import defaultdict


def normalized_sigmoid(x, *, width=1, center=1):
    s = 1 / (1+np.exp(width*(x-center)))
    s = s - np.min(s)
    return s / np.max(s)  # normalize function to 0-1


def show(stimulus, orthographizer, wavesom, sap, depth=5, num=3):
    """
    Show the response of the map to some stimulus.

    :param stimulus: The input stimulus as a string.
    :param orthographizer: The orthographizer used to encode the input stimulus.
    :param wavesom: The trained wavesom.
    :param sap: The word2idx dictionary, for visualization.
    :param depth: The depth to descend in the tree.
    :param num: The number of neighbors to consider in the computation.
    :return: None
    """
    # Imports
    from wavesom.visualization.simple_viz import show_label_activation_map
    import matplotlib.pyplot as plt

    # Close previous plot
    plt.close()

    # Transform the stimulus to a vector.
    a = transfortho(stimulus, orthographizer, wavesom.orth_len)
    # Show the map's response to the stimulus
    show_label_activation_map(sap, wavesom.map_dimensions[0], wavesom.activate_state(a, max_depth=depth, num=num).reshape(wavesom.map_dimensions).T)


def evaluate(pas_dict, s, o):

    results = []
    words = []

    for k, v in pas_dict.items():
        words.append(k)
        results.append(s.predict_part(transfortho(k, o, orth_vec_len), 0)[0:] in v)

    return results, words


def dist_to_lexicon(vec, lexicon, X, num_to_return=5):

    lex = np.array(lexicon)
    dists = np.sum(np.square(X - vec), axis=1)
    s = np.argsort(dists)[:num_to_return]

    return lex[s], dists[s]


def transfortho(x, o, length):

    zero = np.zeros((length,))
    vec = o.vectorize(x).ravel()
    zero[:len(vec)] += vec

    return zero


class Wavesom(Som):

    def __init__(self,
                 map_dim,
                 dim,
                 learning_rate,
                 orth_len,
                 phon_len,
                 lrfunc=expo,
                 nbfunc=expo,
                 sigma=None,
                 min_max=np_min):
        """
        initialize a Wavesom
        """
        super().__init__(map_dim,
                         dim,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         sigma,
                         min_max)

        self.orth_len = orth_len
        self.phon_len = phon_len

    @classmethod
    def load(cls, path, array_type=np):
        """
        Load a Wavesom.

        :param path: The path to the JSON file where the wavesom is stored
        :param array_type: The array type to use.
        :return: A wavesom.
        """
        data = json.load(open(path))

        weights = data['weights']
        weights = array_type.asarray(weights, dtype=np.float32)
        datadim = weights.shape[1]

        dimensions = data['dimensions']
        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear
        lr = data['lr']
        sigma = data['sigma']
        orth_len = data['orth_len']
        phon_len = data['orth_len']

        s = cls(dimensions,
                datadim,
                lr,
                orth_len,
                phon_len,
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                sigma=sigma)

        s.weights = weights
        s.trained = True

        return s

    def save(self, path):
        """
        Save a wavesom.

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
        Compute the prediction for part of the weights, specified by an offset.

        :param X: The input data
        :param offset: The offset which is applied to axis 1 of X
        :return: An matrix containing the distance of each sample to
        each weight.
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        datadim = X.shape[-1]
        X = self._create_batches(X, batch_size=1)

        temp_weights = self.weights[:, offset:offset+datadim]
        distances = []

        for x in X:

            distance = self.distance_function(x, weights=temp_weights)[0]
            distances.extend(distance)

        return np.array(distances)

    def predict_part(self, X, offset, orth_vec_len=0):
        """
        Predict BMUs based on part of the weights.

        :param X:
        :param offset:
        :param orth_vec_len:
        :return:
        """
        if orth_vec_len:
            X = X[:, :orth_vec_len]

        dist = self._predict_base_part(X, offset)
        return self.min_max(dist, axis=1)[1]

    def activate(self, x, max_depth=4, threshold=0.15):

        ortho_act = self._predict_base_part(x, 0)

        normalized_activation = np.squeeze(normalized_sigmoid(ortho_act))

        pre_calc = self._predict_base(self.weights)

        result = defaultdict(list)
        result[0] = [normalized_activation]
        self._inner_activation(normalized_activation, 1, pre_calc, max_depth, threshold, result)

        return np.array([np.array(result[idx]) for idx in sorted(result.keys())])

    def _inner_activation(self, weights, depth, pre_calc, max_depth, threshold, result):

        if depth > 5:
            mixin = False

        for idx in np.flatnonzero(weights > threshold):

            pred = pre_calc[idx]
            res = np.squeeze(normalized_sigmoid(pred) * weights[idx])
            result[depth].append(res)
            if np.any(res > threshold) and depth < max_depth:
                self._inner_activation(res, depth+1, pre_calc, max_depth, threshold, result)

        return result
