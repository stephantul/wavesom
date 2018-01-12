"""Code for the Time-dependent Wavesom."""
import numpy as np

from tqdm import tqdm
from somber import Som


def softmax(w):
    """Calculate the softmax."""
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)[:, None]
    return dist


class Wavesom(Som):
    """
    A Time-Dependent Self-Organizing Map.

    The WaveSom is a Self-Organizing Map equipped with a softmax function.
    This allows the SOM to produce a probability distribution over neurons in
    response to some input. In addition, the WaveSOM is equipped with functions
    that allow it to provide responses to partial input. This allows it to
    account for missing datapoints.

    Parameters
    ==========
    map_dimensions : tuple of int
        The dimensions of the map. Can be any number of dimensions, as the
        underlying library can handle hypermaps.
    data_dimensionality : int
        The number of features.
    learning_rate : float
        The learning rate.
    influence : float, optional, default None
        The starting influence parameter, also sometimes called Lamdba. If
        this is set to None, it is automatically inferred.
    lr_lambda : float
        Controls the steepness of the exponential function that decreases
        the learning rate.
    nb_lambda : float
        Controls the steepness of the exponential function that decreases
        the neighborhood.

    """

    def __init__(self,
                 map_dimensions,
                 data_dimensionality,
                 learning_rate,
                 influence=None,
                 lr_lambda=2.5,
                 infl_lambda=2.5):
        """Init of a time-dependent SOM."""
        super().__init__(map_dimensions,
                         data_dimensionality,
                         learning_rate,
                         influence)

        self.state = None

    def transform_partial(self,
                          X,
                          offset=0,
                          batch_size=1,
                          show_progressbar=False):
        """
        Compute the prediction for part of the weights, specified by an offset.

        Parameters
        ==========
        X : np.array (N, num_features)
            The input data.
        offset : int, optional, default 0
            The offset with which to offset the input data wrt the weights.
            For example, if the offset is 10, then the 0th column of X will be
            assumed to be aligned with the 10th column of the weights W.
        batch_size : int, optional, default 1
            The batch size to use.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar.

        Returns
        =======
        activations : np.array (N, num_neurons)
            The activation of each neuron in response to each datapoint.

        """
        batched = self._create_batches(X, batch_size, shuffle_data=False)

        activations = []
        temp_weights = self.weights[:, offset:(offset+X.shape[1])]

        for x in tqdm(batched, disable=not show_progressbar):
            activations.extend(self.distance_function(x, temp_weights)[0])

        activations = np.asarray(activations, dtype=np.float32)
        activations = activations[:X.shape[0]]
        return activations.reshape(X.shape[0], self.num_neurons)

    def predict_partial(self, X, offset=0):
        """Predict BMUs based on part of the weights."""
        dist = self.transform_partial(X, offset)
        return dist.__getattribute__(self.argfunc)(axis=1)

    def converge(self, X, max_iter=1000, tol=0.0001, reset_state=True):
        """Run activations until convergence."""
        output = []
        idxes = []

        states = np.zeros((1, self.num_neurons)) / self.num_neurons

        for x in X:
            o = []
            if reset_state:
                states = np.zeros((1, len(self.weights)))
            prev = None

            for idx in range(max_iter):
                states = self.activate(x, states)
                if idx != 0:
                    z = np.abs((states / states.sum()) - (prev / prev.sum())).sum(-1)

                    if np.all((z < tol)):
                        idxes.extend([idx] * len(states))
                        break

                prev = np.copy(states)
                o.append(prev)
            else:
                idxes.extend([idx] * len(states))

            output.append(np.array(o))

        return output, idxes

    def center(self, states):
        """Center the states."""
        s = self.exemplarify(states)
        # Centering
        s -= s.min(0)[None, :]
        # Shifting
        return np.nan_to_num(s / (s.max(0) / self.weights.max(0)))

    def activation_function(self, X):
        """
        Generate an activation given some input.

        The activation function returns an n-dimensional vector between 0
        and 1, where values closer to 1 imply more similarity.

        Parameters
        ==========
        X : np.array (N, num_features)
            The inut data.

        Returns
        =======
        z : np.array (N, num_neurons)
            The activity of the map in response to the input data.

        """
        if np.ndim(X) == 1:
            X = X[None, :]

        z = softmax(-self.transform_partial(X))
        return z

    def exemplarify(self, states):
        """Extract the current state vector as an exemplar."""
        return (self.weights[None, :, :] * states[:, :, None]).mean(1)

    def activate(self, x, states):
        """Activate the network for single iteration."""
        inp = self.activation_function(x[None, :])
        f = self.exemplarify(states)
        delta = self.activation_function(f)
        return (inp + delta) / 2
