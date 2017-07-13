import numpy as np
import json

from somber import Som
from somber.utils import expo, linear, np_min
from collections import defaultdict


def sigmoid(x, *, width=1, center=1):
    return 1 / (1+np.exp(width*(x-center)))

def normalize(x, *, switch=True):

    if switch:
        x = x - np.min(x)
    return x / np.max(x)

def weird_normalize(x):

    x = x / np.max(x)
    return 1 - x

def softmax(x):

    x = np.exp(x - np.max(x))
    return x / x.sum()


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
        self.state = np.random.random(size=len(self.weights)) * .15
        self.cache = None

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
        s.cache = np.array([weird_normalize(sigmoid(x)) for x in s._predict_base(s.weights)])
        [s.activate() for x in range(100)]

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

    def activate(self, x=None):

        if x is None:
            x = np.ones(len(self.weights)) / len(self.weights)
            # x /= len(x)
        else:
            x = sigmoid(self._predict_base_part(x, 0)[0])

        self.state += x
        self.state = np.mean(self.cache * self.state, 0)
        # self.state += s
        # self.state = normalize(self.state, switch=False)
        # self.state.clip(min=0.1)
        return np.copy(self.state), 0
