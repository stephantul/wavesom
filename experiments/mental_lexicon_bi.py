"""Mental lexicon bilingualism."""
import numpy as np
import json
import cProfile
import time

from wavesom.wavesom import Wavesom, softmax
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer, CVTransformer, OpenNGramTransformer
from wordkit.features import binary_features, fourteen
from wordkit.feature_extraction import phoneme_features, one_hot_characters, one_hot_phonemes
from wordkit.samplers import Sampler
from sklearn.pipeline import FeatureUnion
from sklearn.cluster import KMeans
# from skcmeans.algorithms import Probabilistic
from string import ascii_lowercase
from matplotlib import pyplot as plt

from collections import defaultdict


def reconstruction_error(X, reconstruction):
    """Calculates the reconstruction error for binary arrays"""
    return 1 - np.sum(X != reconstruction) / np.prod(X.shape)


def scoring(reconstruction, X, words, idx=0):

    subbed = X[None, :, :] - reconstruction[:, None, :]
    dist = np.linalg.norm(subbed, axis=2).argmin(1)
    res = [(words[d], words[idx]) for idx, d in enumerate(dist)]
    wrong = [x for x in res if (x[0].split()[idx] != x[1].split()[idx])]
    return 1 - (len(wrong) / len(res)), wrong


if __name__ == "__main__":

    wordlist = {'kind', 'mind', 'bind',
                'room', 'wolf', 'way',
                'wee', 'eten',
                'wind', 'lead', 'speed',
                'meat', 'meet', 'meten',
                'spring', 'winter', 'spel',
                'spill', 'spoon',
                'moon', 'boon', 'doen',
                'biet', 'beat', 'spijt',
                'tijd', 'mijt', 'klein',
                'fijn', 'rein', 'reign',
                'feign', 'fine', 'mine',
                'dine', 'wine', 'wijn',
                'weinig', 'meel',
                'keel', 'veel', 'kiel',
                'wiel', 'wheel', 'feel',
                'deal', 'spool', 'spoel',
                'trol', 'troll', 'wolf',
                'rond', 'mond', 'mound',
                'hound', 'found', 'find',
                'spine', 'mine',
                'cake', 'leem',
                'dessert', 'desert', 'model',
                'curve', 'dies', 'nies', 'vies',
                'boot', 'potent', 'sate',
                'tree', 'rapport', 'lied',
                'mond', 'kei',
                'steen', 'geen',
                'leek', 'gek',
                'creek', 'ziek', 'piek'}

    o = LinearTransformer(one_hot_characters(ascii_lowercase))
    p = ONCTransformer(one_hot_phonemes(use_long=True))

    transformers = FeatureUnion([("o", o), ("p", p)])

    f = lambda x: x['syllables']

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True, filter_function=f)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True, filter_function=f)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])

    np.random.seed(21)

    words_ = corpora.fit_transform(wordlist)
    X = transformers.fit_transform(words_)

    samp = Sampler(X, words_, [x['frequency'] for x in words_], mode='log')
    X_sampled, _ = samp.sample(10000)
    o_len = o.vec_len

    words = [" ".join((x['orthography'], "-".join(["|".join(y) for y in x['syllables']]))) for x in words_]

    s = Wavesom((5, 5), X.shape[1], 1.0)
    s.fit(X_sampled, num_epochs=10, show_progressbar=True, batch_size=1)

    # s = Wavesom.load("saved_models/mental_lexicon_bi.json")
    # print("Training finished")

    w2l = defaultdict(list)
    w2l_form = defaultdict(list)
    l2w = defaultdict(set)
    quant = {}
    w2id = defaultdict(list)

    for idx, (n, x, q) in enumerate(zip(words, s.predict(X), s.quantization_error(X))):
        w2id[n.split()[0]].append(idx)
        w2l_form[n.split()[0]].append(x)
        w2l[n].append(x)
        l2w[x].add(n)
        quant[n] = q

    scores = {}

    results = []
    states = []

    start = time.time()

    # baseline
    p = s.predict_partial(X[:, :o_len], 0)
    inv = s.invert_projection(X, words).reshape(s.num_neurons)

    baseline = []
    for idx, x in enumerate(p):
        baseline.append([words[idx], inv[x]])

    baseline_score = len([x for x, y in baseline if (x.split()[0] == y.split()[0])]) / len(baseline)

    soft_part = s.activation_function(X[:, :o_len])
    soft_full = s.activation_function(X)

    rep_part = s.center(soft_part)
    rep_full = s.center(soft_full)

    soft_test_part = softmax(-s.transform_partial(X[:, :o_len]))
    soft_test_full = softmax(-s.transform_partial(X))

    rep_test_part = s.center(soft_test_part)
    rep_test_full = s.center(soft_test_full)

    softmax_reconstruction, _ = scoring(rep_part, X, words)
    softmax_reconstruction_phon, _ = scoring(rep_part[:, o_len:], X[:, o_len:], words, idx=1)
    softmax_score, _ = scoring(rep_part, rep_full, words)

    test_1 = scoring(rep_test_part, rep_test_full, words)
    test_2 = scoring(rep_test_part, X, words)
    test_3 = scoring(rep_test_part[:, o_len:], X[:, o_len:], words, idx=1)

    print("Took {} seconds".format(time.time() - start))
    print("Soft: ", softmax_score)
    print("Soft_orth: ", softmax_reconstruction)
    print("Soft_phon ", softmax_reconstruction_phon)
    print("Base: ", baseline_score)
