import json
import numpy as np

from wavesom.wavesom import Wavesom, transfortho, show, normalized_sigmoid
from experiments.setup import setup
from experiments.read_blp import read_blp_format

from collections import defaultdict


if __name__ == "__main__":

    path = "saved_models/bilingual_model_big.json"

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

    X, X_orig, s_ = setup(dicto, max_len, wordlist)
    orth_vec_len = 14 * max_len

    s = Wavesom.load(path, np)

    w2l = defaultdict(list)
    l2w = defaultdict(set)
    quant = {}

    for n, x, q in zip(s_, s.predict(X_orig), s.quant_error(X_orig)):
        n = " ".join([n[0], "-".join(n[1])])
        w2l[n].append(x)
        l2w[x].add(n)
        quant[n] = q

    unique_ortho = {v[0]: X_orig[idx, :orth_vec_len] for idx, v in enumerate(s_)}

    words, vectors = zip(*unique_ortho.items())
    vectors = np.array(vectors)

    predicted_orthography = s._predict_base_part(vectors, 0)
    sorted_predictions = predicted_orthography.argsort()

    values = []

    for idx, x in enumerate(sorted_predictions):
        values.append(predicted_orthography[idx, x[:5]])

    x = dict(zip(*[words, values]))

    rt_data = dict(read_blp_format(filename="data/blp-items.txt", words=wordlist))

    error_rt_vector = []
    for k, v in x.items():

        try:
            rt = rt_data[k.lower()]
        except KeyError:
            print("{} not in RT".format(k))

        error_rt_vector.append((v[0], rt))

    a, b = error_rt_vector = np.array(error_rt_vector).T

    import matplotlib.pyplot as plt
    import cProfile

    # a, b = [y for x, y in w2l.items() if x.startswith("keel")]
    # w1 = s.weights[a, :]
    # w2 = s.weights[b, :]

    # p = transfortho("KEEL", o, s.weights.shape[1])
    # dist1 = np.linalg.norm(p[None, None, :] - w1[None, :, :])
    # dist2 = np.linalg.norm(p[None, None, :] - s.weights[:, None, :], axis=2)

    # inv = {idx: k for idx, k in enumerate(s.invert_projection(X_orig, unique_words))}
    # from wavesom.visualization.simple_viz import show_labelmap
    # show_labelmap(inv, 50)
