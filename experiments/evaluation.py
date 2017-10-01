"""Various evaluation scripts."""
import numpy as np


def evaluate_symbolic(s, words, vectors, inverted_mapping):
    """
    Evaluate the SOM by assuming every neurom stores one word.

    :param s: The som to evaluate.
    :param words: The words to evaluate.
    :param vectors: The vectors of the corresponding words.
    :param inverted_mapping: An inverted mapping from neurons to words.
    :return: A list of tuples containing the stimuli and predicted words for
    the presented stimuli.
    """
    results = []
    indices = s.predict_part(vectors, 0)

    for w, x in zip(words, indices):

        w_2 = inverted_mapping[x].split()[0]
        results.append((w, w_2))

    return results

def evaluate_distributed(states_full, states_orthographic, words, o_words):

    best_match = []

    for x in range(0, len(states_orthographic), 250):
        sub = states_orthographic[x:x+250, None, :] - states_full[None, :, :]
        best_match.extend(np.linalg.norm(sub, axis=2).argmin(1))

    results = []

    for idx, x in enumerate(best_match):
        results.append((o_words[idx], words[x].split()[0]))

    return results
