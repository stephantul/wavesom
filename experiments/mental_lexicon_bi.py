"""Mental lexicon bilingualism."""
import numpy as np
import json
import cProfile
import time

from wavesom.wavesom import Wavesom
# from wavesom.wavesom_old import Wavesom
from wavesom.wavesom_older import Wavesom as ws_old
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer
from wordkit.features import fourteen, binary_features
from wordkit.feature_extraction import phoneme_features
from sklearn.pipeline import FeatureUnion

from collections import defaultdict


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

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])

    words_ = corpora.fit_transform(wordlist)
    X = transformers.fit_transform(words_)
    X[X == 0] = -1.
    words = [" ".join((x['orthography'], "-".join(x['syllables']))) for x in words_]

    np.random.seed(44)

    s = Wavesom.load(path, np)

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

    inv = s.invert_projection(X, words)
    inv_o = s.invert_projection(vectors, o_words).reshape(s.weight_dim)

    scores = {}

    results = []
    states = []

    start = time.time()
    p, idx = s.converge(X, max_iter=10000, batch_size=1)
    z, idx = s.converge(vectors[:, :s.orth_len], max_iter=10000, batch_size=1)
    print("Took {} seconds".format(time.time() - start))

    pert = np.linalg.norm(p[None, :, :] - z[:, None, :], axis=2).argmin(1)
    res_2 = [(o_words[idx], words[d]) for idx, d in enumerate(pert)]
    score_2 = len([x for x in res_2 if (x[0] == x[1].split(" ")[0])]) / len(res_2)

    zert = s.statify(z)
    # escore = len([x for x in np.all(z[None, :, :] - X[:, None, :], 1) if x]) / len(X)
    dist = np.linalg.norm(zert[None, :, :] - X[:, None, :], axis=2).argmin(0)
    res = [(o_words[idx], words[d]) for idx, d in enumerate(dist)]
    score = len([x for x in res if (x[0] == x[1].split(" ")[0])]) / len(res)
