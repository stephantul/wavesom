"""Mental lexicon bilingualism."""
import numpy as np
import json
import cProfile
import time

from wavesom.wavesom import Wavesom
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer
from wordkit.features import binary_features, fourteen
from wordkit.feature_extraction import phoneme_features, one_hot_characters, one_hot_phonemes
from sklearn.pipeline import FeatureUnion
from sklearn.cluster import KMeans
# from skcmeans.algorithms import Probabilistic
from string import ascii_lowercase
from matplotlib import pyplot as plt

from collections import defaultdict


def reconstruction_error(X, reconstruction):
    """Calculates the reconstruction error for binary arrays"""
    return 1 - np.sum(X != reconstruction) / np.prod(X.shape)


def scoring(reconstruction, X, words):

    subbed = X[None, :, :] - reconstruction[:, None, :]
    dist = np.linalg.norm(subbed, axis=2).argmin(1)
    res = [(words[d], words[idx]) for idx, d in enumerate(dist)]
    wrong = [x for x in res if (x[0].split(" ")[0] != x[1].split(" ")[0])]
    return 1 - (len(wrong) / len(res)), wrong


if __name__ == "__main__":

    path = "saved_models/amlap_lex_1000e_1010map.json"

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

    o = LinearTransformer(fourteen)
    p = ONCTransformer(phoneme_features(binary_features))

    transformers = FeatureUnion([("o", o), ("p", p)])

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True, minfreq=0)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True, minfreq=0)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])

    np.random.seed(21)

    words_ = corpora.fit_transform(wordlist)
    # words_ = [x for x in words_ if not set(x['orthography']) - set(ascii_lowercase) and len(x['orthography']) <= 6]
    new_words = []
    for x in np.random.randint(0, len(words_), 1000):
        new_words.append(words_[x])
    words_ = new_words
    X = transformers.fit_transform(words_)
    # X = X[:, X.sum(0) != 0]

    words = [" ".join((x['orthography'], "-".join(x['syllables']))) for x in words_]

    num_clust = 100

    # k = KMeans(num_clust)
    # k.fit(X)
    s = Wavesom.load(path)
    s.weights = (s.weights + 1) * .5
    # s = Wavesom((num_clust, 1), X.shape[1], 1.0, dampening=.1)
    # s.weights = k.cluster_centers_

    print("Training finished")
    # s = Wavesom.load(path, np)

    w2l = defaultdict(list)
    w2l_form = defaultdict(list)
    l2w = defaultdict(set)
    quant = {}
    w2id = defaultdict(list)

    for idx, (n, x, q) in enumerate(zip(words, s.predict(X), s.quant_error(X))):
        w2id[n.split()[0]].append(idx)
        w2l_form[n.split()[0]].append(x)
        w2l[n].append(x)
        l2w[x].add(n)
        quant[n] = q

    scores = {}

    results = []
    states = []

    start = time.time()

    scores = {}

    num_stuff = 100

    for idx, x in enumerate(np.linspace(-.8, .8, num_stuff)):

        print("{}/{}".format(idx, num_stuff))
        s.dampening = x
        print(s.dampening)
        start_2 = time.time()

        full_conv, full_idx = s.converge(X)
        partial_conv, partial_idx = s.converge(X[:, :o.vec_len])

        conv_score, _ = scoring(partial_conv, full_conv, words)

        conv_reconstruction = s.center(partial_conv)

        # Calculate scores
        conv_score_reconstruction, _ = scoring(conv_reconstruction, X, words)
        conv_score_reconstruction_phon, _ = scoring(conv_reconstruction[:, o.vec_len:], X[:, o.vec_len:], words)

        scores[x] = (conv_score, conv_score_reconstruction, conv_score_reconstruction_phon, np.max(partial_idx), np.mean(partial_idx), np.min(partial_idx))

    print("Took {} seconds".format(time.time() - start_2))

    print("Took {} seconds".format(time.time() - start))

    # baseline
    p = s.predict_part(X[:, :o.vec_len], 0)
    inv = s.invert_projection(X, words).reshape(s.weight_dim)

    baseline = []
    for idx, x in enumerate(p):
        baseline.append([words[idx], inv[x]])

    baseline_score = len([x for x, y in baseline if (x.split()[0] == y.split()[0])]) / len(baseline)

    soft_all = s.activation_function(X)
    rep_a = s.center(soft_all)

    softmax_reconstruction, _ = scoring(rep_a, X, words)

    soft_part = s.activation_function(X[:, :o.vec_len])
    rep_b = s.center(soft_part)

    softmax_score, _ = scoring(rep_b, rep_a, words)

    print("Took {} seconds".format(time.time() - start))
    print("Soft: ", softmax_score)
    print("S_RE: ", softmax_reconstruction)
    print("Conv: ", conv_score)
    print("Reco: ", conv_score_reconstruction)
    print("Phon: ", conv_score_reconstruction_phon)
    print("Base: ", baseline_score)

    xos = list(zip(*sorted(scores.items(), key=lambda x: x[0])))[1]
    a, b, c, _, _, _ = zip(*xos)

    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
