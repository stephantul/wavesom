import json
import numpy as np

from wavesom.wavesom import Wavesom
from wavesom.experiments.setup import setup
from wavesom.experiments.preprocessing.orthographizer.mcclelland import mcclelland_orthography

from collections import defaultdict


def show(stimulus, orthographizer, wavesom, sap, depth=5, num=3):
    """
    Shows the response of the map to some stimulus

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
    show_label_activation_map(sap, 10, wavesom.activate_state(a, max_depth=depth, num=num).reshape(10, 10).T)


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

if __name__ == "__main__":

    path = ""
    dicto = json.load(open("data/syllable_lexicon.json"))

    max_len = 7

    wordlist = {'kind', 'mind', 'bind',
                'room', 'wolf', 'way',
                'wee', 'eten', 'eating',
                'wind', 'lead', 'speed',
                'meat', 'meet', 'meten',
                'spring', 'winter', 'spel',
                'speel', 'spill', 'spoon',
                'moon', 'boon', 'doen',
                'biet', 'beat', 'spijt',
                'tijd', 'mijt', 'klein',
                'fijn', 'rein', 'reign',
                'feign', 'fine', 'mine',
                'dine', 'wine', 'wijn',
                'weinig', 'speel', 'meel',
                'keel', 'veel', 'kiel',
                'wiel', 'wheel', 'feel',
                'deal', 'spool', 'spoel',
                'trol', 'troll', 'wolf',
                'rond', 'mond', 'mound',
                'hound', 'found', 'find',
                'lined', 'spine', 'mine',
                'cake', 'keek', 'leem',
                'dessert', 'desert', 'model',
                'curve', 'dies', 'nies', 'vies',
                'boot', 'potent', 'sate',
                'tree', 'rapport', 'lied',
                'bied', 'mond', 'kei',
                'steen', 'meen', 'geen',
                'keek', 'leek', 'gek',
                'creek', 'ziek', 'piek'}

    o = mcclelland_orthography()

    X, X_orig, s_, c = setup(dicto, max_len, wordlist)
    orth_vec_len = 14 * max_len

    s = Wavesom.load(path, np)

    w2l = defaultdict(list)
    l2w = defaultdict(list)

    for n, x in zip(s_, s.predict(X_orig)):
        w2l[n].append(x)
        l2w[x].append(n)
