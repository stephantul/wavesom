import numpy as np
import logging
import time

from somplay.som import Som
from functools import reduce
from somplay.utils import progressbar
from somplay.experiments.markov_chain import MarkovGenerator

logger = logging.getLogger(__name__)


def sigmoid_array(x):

    return 1 / (1 + np.exp(-x))


class Feedforward(object):

    def __init__(self, soms=()):

        self._check_dim(soms)
        self.soms = soms

    def predict(self, X):

        pred = X

        for som in self.soms[:-1]:
            pred = som._predict_base(pred)

        return self.soms[-1].predict(pred)

    @staticmethod
    def _check_dim(soms):

        for idx, som in enumerate(soms[:-1]):

            if not som.weights.shape[0] == soms[idx+1].data_dim:
                raise ValueError("Soms {0} and {1} do not have the correct dimensionality")

    def train(self, X, num_epochs, total_updates=1000, stop_lr_updates=1.0, stop_nb_updates=1.0, context_mask=(), show_progressbar=False):
        """
        Fits the SOM to some data.
        The updates correspond to the number of updates to the parameters
        (i.e. learning rate, neighborhood, not weights!) to perform during training.

        In general, 1000 updates will do for most learning problems.

        :param X: the data on which to train.
        :param total_updates: The number of updates to the parameters to do during training.
        :param stop_lr_updates: A fraction, describing over which portion of the training data
        the neighborhood and learning rate should decrease. If the total number of updates, for example
        is 1000, and stop_updates = 0.5, 1000 updates will have occurred after half of the examples.
        After this period, no updates of the parameters will occur.
        :param context_mask: a binary mask used to indicate whether the context should be set to 0
        at that specified point in time. Used to make items conditionally independent on previous items.
        Examples: Spaces in character-based models of language. Periods and question marks in models of sentences.
        :return: None
        """

        train_length = len(X) * num_epochs

        # The step size is the number of items between rough epochs.
        # We use len instead of shape because len also works with np.flatiter
        step_size_lr = max((train_length * stop_lr_updates) // total_updates, 1)
        step_size_nb = max((train_length * stop_nb_updates) // total_updates, 1)

        if step_size_lr == 1 or step_size_nb == 1:
            logger.warning("step size of learning rate or neighborhood is 1")

        logger.info("{0} items, {1} epochs, total length: {2}".format(len(X), num_epochs, train_length))

        # Precalculate the number of updates.
        lr_update_counter = np.arange(step_size_lr, (train_length * stop_lr_updates) + step_size_lr, step_size_lr)
        nb_update_counter = np.arange(step_size_nb, (train_length * stop_nb_updates) + step_size_nb, step_size_nb)
        start = time.time()

        # Train
        self._train_loop(X, num_epochs, lr_update_counter, nb_update_counter, context_mask, show_progressbar)
        self.trained = True

        logger.info("Total train time: {0}".format(time.time() - start))

    def _train_loop(self, X, num_epochs, nb_update_counter, lr_update_counter, context_mask, show_progressbar):

        nb_step = 0
        lr_step = 0

        map_radii = [som.nbfunc(som.sigma, nb_step, len(nb_update_counter)) for som in self.soms]
        learning_rates = [som.lrfunc(som.learning_rate, lr_step, len(lr_update_counter)) for som in self.soms]
        influences = [som._calculate_influence(map_radius) for som, map_radius in zip(self.soms, map_radii)]
        update = False

        idx = 0

        for epoch in range(num_epochs):

            for x in progressbar(X, use=show_progressbar):

                activation = self.soms[0]._example(x, influences[0])

                for som, influence in zip(self.soms[1:], influences[1:]):
                    if np.any(np.isnan(activation)):
                        raise ValueError("NAN")
                    activation = som._example(activation, influence)

                if idx in nb_update_counter:
                    nb_step += 1

                    map_radii = [som.nbfunc(som.sigma, nb_step, len(nb_update_counter)) for som in self.soms]

                if idx in lr_update_counter:
                    lr_step += 1

                    learning_rates = [som.lrfunc(som.learning_rate, lr_step, len(lr_update_counter)) for som in
                                      self.soms]
                    update = True

                if update:

                    influences = [sm[0]._calculate_influence(sm[1]) * learning_rates[idx] for idx, sm in enumerate(zip(self.soms, map_radii))]
                    update = False

                idx += 1


if __name__ == "__main__":

    soms = [Som((10, 10), 4, 1.0), Som((5, 5), 100, 1.0), Som((3, 3), 25, 1.0), Som((2, 2), 9, 1.0)]

    b = Feedforward(soms)

    m = MarkovGenerator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.array([[0.25, 0.25, 0.25, 0.25] * 4]).reshape(4, 4), [0.25, 0.25, 0.25, 0.25])
    X = m.generate_sequences(1, 10000)[0]
    print(X.shape)

    # X = np.random.binomial(1, 0.5, (10000,))[:, np.newaxis]

    b.train(X, 10, show_progressbar=True)
    print("done")
