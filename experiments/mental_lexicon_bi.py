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

from collections import defaultdict


def reconstruction_error(X, loek):

    return 1 - np.sum(X != loek) / np.prod(X.shape)


def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)[:, None]
    return dist


def soft_clip(X, weights, length=None, normalize=False):

    if length is None:
        length = X.shape[-1]
    dists = np.linalg.norm(X[:, None, :length] - weights[None, :, :length],
                           axis=2)
    if normalize:
        # dists -= dists.mean(0)
        dists /= dists.sd(0)
    return np.clip(dists.mean(1)[:, None] - dists, a_min=0, a_max=None)


def max_dist_clipping(X, weights, length=None):

    if length is None:
        length = X.shape[-1]
    dists = np.linalg.norm(X[:, None, :length] - weights[None, :, :length],
                           axis=2)

    return dists

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

    o = LinearTransformer(one_hot_characters(ascii_lowercase))
    p = ONCTransformer(one_hot_phonemes())

    transformers = FeatureUnion([("o", o), ("p", p)])

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True, minfreq=0)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True, minfreq=0)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])

    np.random.seed(68)

    words_ = corpora.fit_transform([])
    words_ = [x for x in words_ if not set(x['orthography']) - set(ascii_lowercase) and len(x['orthography']) <= 6]
    new_words = []
    for x in np.random.randint(0, len(words_), 100):
        new_words.append(words_[x])
    words_ = new_words
    X = transformers.fit_transform(words_)
    X = X[:, X.sum(0) != 0]

    words = [" ".join((x['orthography'], "-".join(x['syllables']))) for x in words_]

    num_clust = 10

    k = KMeans(num_clust)
    k.fit(X)
    # s = Wavesom.load(path)
    # s.weights = (s.weights + 1) * .5
    s = Wavesom((num_clust, 1), X.shape[1], 1.0)
    s.weights = k.cluster_centers_

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

    unique_ortho = {v.split()[0]: X[idx] for idx, v in enumerate(words)}

    o_words, vectors = zip(*unique_ortho.items())
    vectors = np.array(vectors)

    scores = {}

    results = []
    states = []

    start = time.time()

    scores = {}

    full_conv, full_idx = s.converge(X, max_iter=10000, batch_size=1)
    partial_conv, partial_idx = s.converge(vectors[:, :o.vec_len], max_iter=10000, batch_size=1)

    conv = np.linalg.norm(full_conv[None, :, :] - partial_conv[:, None, :], axis=2).argmin(1)
    res_conv = [(o_words[idx], words[d]) for idx, d in enumerate(conv)]
    conv_score = len([x for x in res_conv if (x[0].split(" ")[0] == x[1].split(" ")[0])]) / len(res_conv)

    conv_reconstruction = s.statify(partial_conv)
    conv_reconstruction -= conv_reconstruction.min(0)[None, :]
    conv_reconstruction = conv_reconstruction / (conv_reconstruction.max(0) / X.max(0))
    conv = np.linalg.norm(X[None, :, :] - conv_reconstruction[:, None, :], axis=2).argmin(1)
    res_reco = [(o_words[idx], words[d]) for idx, d in enumerate(conv)]
    conv_score_reconstruction = len([x for x in res_reco if (x[0].split(" ")[0] == x[1].split(" ")[0])]) / len(res_reco)

    scores[x] = (conv_score, conv_score_reconstruction, np.max(partial_idx), np.mean(partial_idx), np.min(partial_idx))

    # baseline
    p = s.predict_part(vectors[:, :o.vec_len], 0)
    inv = s.invert_projection(X, words).reshape(s.weight_dim)

    baseline = []
    for idx, x in enumerate(p):
        baseline.append([o_words[idx], inv[x]])

    baseline_score = len([x for x, y in baseline if (x == y.split()[0])]) / len(baseline)

    z = 1 / np.square(s.predict_distance_part(vectors[:, :o.vec_len] + 10e-5, 0))
    z **= 2
    peh = z / z.sum(1)[:, None]
    rep = (s.weights[None, :, :] * peh[:, :, None]).sum(1)
    dist = np.linalg.norm(rep[None, :, :] - X[:, None, :], axis=2).argmin(0)
    res = [(o_words[idx], words[d]) for idx, d in enumerate(dist)]
    dist_score_3 = len([x for x in res if (x[0].split(" ")[0] == x[1].split(" ")[0])]) / len(res)

    z = s.predict_distance_part(vectors[:, :o.vec_len], 0)
    z = softmax(z.max(1)[:, None] - z)
    rep = (s.weights[None, :, :] * z[:, :, None]).sum(1)
    rep -= rep.min(0)[None, :]
    rep = rep / (rep.max(0) / X.max(0))
    dist = np.linalg.norm(rep[None, :, :] - X[:, None, :], axis=2).argmin(0)
    res = [(o_words[idx], words[d]) for idx, d in enumerate(dist)]
    softmax_score = len([x for x in res if (x[0].split(" ")[0] == x[1].split(" ")[0])]) / len(res)

    print("Took {} seconds".format(time.time() - start))
    print("Soft: ", softmax_score)
    print("Dist: ", dist_score_3)
    print("Conv: ", conv_score)
    print("Reco: ", conv_score_reconstruction)
    print("Base: ", baseline_score)
