"""Code for the Time-dependent Wavesom."""
import numpy as np
import json
import cupy as cp
from tqdm import tqdm

from somber import Som
from somber.components.utilities import expo


class Wavesom(Som):
    """A Time-Dependent SOM."""

    # Static property names
    param_names = {'neighborhood',
                   'learning_rate',
                   'map_dimensions',
                   'weights',
                   'data_dimensionality',
                   'lrfunc',
                   'nbfunc',
                   'valfunc',
                   'argfunc',
                   'orth_len',
                   'phon_len'}

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 orth_len,
                 phon_len,
                 lrfunc=expo,
                 nbfunc=expo,
                 neighborhood=None):

        super().__init__(map_dimensions,
                         data_dimensionality,
                         learning_rate,
                         lrfunc,
                         nbfunc,
                         neighborhood)

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

        lrfunc = expo if data['lrfunc'] == 'expo' else linear
        nbfunc = expo if data['nbfunc'] == 'expo' else linear

        s = cls(data['map_dimensions'],
                data['data_dimensionality'],
                data['learning_rate'],
                data['orth_len'],
                data['phon_len'],
                lrfunc=lrfunc,
                nbfunc=nbfunc,
                neighborhood=data['neighborhood'])

        s.weights = weights
        s.trained = True

        return s

    def predict_distance_part(self,
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

        dist = self.predict_distance_part(X, offset)
        return dist.__getattr__(self.argfunc)(axis=1)

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
        x = np.exp(-np.squeeze(self.predict_distance_part(x[None, :], 0)))
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
            output.append(np.squeeze(s))
        output = np.array(output)
        if output.ndim == 1:
            return output[None, :]
        return output

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
            x = self.activation_function(x)

        output = []

        for idx in range(iterations):

            p = self.activation_function(self.statify())
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
