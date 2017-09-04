"""Code for the Time-dependent Wavesom."""
import numpy as np
import json
import cupy as cp
from tqdm import tqdm

from somber import Som
from somber.utils import expo, linear, np_min


class Wavesom(Som):
    """A Time-Dependent SOM."""

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

        super().__init__(map_dim,
                         dim,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         sigma,
                         min_max)

        self.orth_len = orth_len
        self.phon_len = phon_len
        self.state = np.ones(len(self.weights))

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

    def _predict_base_part(self,
                           X,
                           offset,
                           batch_size=1,
                           show_progressbar=False):
        """
        Compute the prediction for part of the weights, specified by an offset.

        :param X: The input data
        :param offset: The offset which is applied to axis 1 of X before
        calculating similarity.
        :return: A matrix containing the distance of each sample to
        each weight.
        """
        xp = cp.get_array_module()
        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []
        temp_weights = self.weights[:, offset:(offset+X.shape[1])]

        for x in tqdm(batched, disable=not show_progressbar):
            activations.extend(self.distance_function(x, temp_weights)[0])

        activations = xp.asarray(activations, dtype=xp.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.weight_dim)

    def predict_part(self, X, offset, vec_length=0):
        """
        Predict BMUs based on part of the weights.

        :param X: The input data.
        :param offset: The offset which is applied to axis 1 of X before
        calculating similarity.
        :param orth_vec_len: The length of the vector over which similarity
        is calculated.
        :return:
        """
        if vec_length:
            X = X[:, :vec_length]

        dist = self._predict_base_part(X, offset)
        return self.min_max(dist, axis=1)[1]

    def statify(self):
        """Extract the current state vector as an exemplar."""
        p = (self.weights * self.state[:, None]).mean(0)
        return p

    def activation_function(self, x):
        """
        Generate an activation given some input.

        The activation function returns an n-dimensional vector between 0
        and 1, where values closer to 1 imply more similarity.

        :param x: The input datum.
        :return: An activation.
        """
        x = np.exp(-np.squeeze(self._predict_base_part(x[None, :], 0)))
        x -= (x.mean() + x.std())
        return x

    def converge(self, x, max_iter=1000, tol=0.001):
        """
        Run activations until convergence.

        Convergence is specified as the point when the difference between
        the state vector in the current step and the previous step is closer
        than the tolerance.

        :param x: The input.
        :param max_iter: The maximum iterations to run for.
        :param tol: The tolerance threshold.
        :return: A 2D array, containing the states the system moved through
        while converging.
        """
        output = []

        for idx in range(max_iter):
            s = self.activate(x, iterations=1)
            if idx != 0 and np.abs(np.sum(s[0] - output[-1])) < tol:
                break
            output.append(s)
        if np.array(output).ndim == 2:
            return output
        return np.squeeze(output)

    def activate(self, x=None, iterations=20):
        """
        Activate the network for a number of iterations.

        :param x: The input, can be None, in which case the systrm oscillates.
        :param iterations: The number of iterations for which to run.
        :return: A 2D array, containing the states the system moved through
        """
        if x is None:
            x = np.zeros((len(self.weights)))
        else:
            x = self.activate(x)

        output = []

        for idx in range(iterations):

            p = self.activate(self.statify())
            delta = x + p
            pos = delta >= 0
            neg = delta < 0

            # The ceiling is set at 2.0
            # This term ensures that updates get smaller as
            # activation approaches the ceiling.
            ceiling = (1.0 - (self.state[pos] / 2.))
            # Do dampening.
            self.state[pos] += delta[pos] * ceiling
            self.state[neg] += delta[neg] * self.state[neg]
            output.append(np.copy(self.state))

        return np.array(output)
