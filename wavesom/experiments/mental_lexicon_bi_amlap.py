import json
import numpy as np
import time

from somplay.amlap.lexisom import Lexisom
from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.experiments.preprocessing.bow_encoder import BowEncoder
from somplay.experiments.preprocessing.sivi.ipapy_features import ipapy_sivi
from somplay.experiments.preprocessing.sampler import random_sample

from collections import Counter, defaultdict
from string import ascii_lowercase


def show(word, o, orth_vec_len, s, sap, depth=5, num=3):

    from somplay.visualization.simple_viz import show_label_arrow_map, show_label_scatter_map, show_label_activation_map
    import matplotlib.pyplot as plt

    plt.close()
    plt.figure()

    a = transfortho(word, o, orth_vec_len)
    b = s.propagate(a, max_depth=depth, num=num)

    # show_label_arrow_map(sap, 10, b)
    show_label_scatter_map(sap, 10, b)
    show_label_activation_map(sap, 10, s.get_value(a, max_depth=depth, num=num).reshape(10, 10).T)


def evaluate(pas_dict, s):

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
    vec = o.transform(x).ravel()
    zero[:len(vec)] += vec

    return zero

if __name__ == "__main__":

    english_dict = json.load(open("data/english.json"))
    dutch_dict = json.load(open("data/dutch.json"))

    eng_id = {"{0} {1}".format(o, p) for o, _, p in english_dict}
    dutch_id = {"{0} {1}".format(o, p) for o, _, p in dutch_dict}

    lang_dict = {}

    for e in eng_id.union(dutch_id):
        if e in eng_id and e in dutch_id:
            lang_dict[e] = 'both'
        elif e in eng_id:
            lang_dict[e] = 'eng'
        else:
            lang_dict[e] = 'dut'

    max_len = 7
    max_lex = 1000

    sivi = ipapy_sivi(3)
    o = Orthographizer()
    orth_vec_len = max_len * o.datalen

    np.random.seed(44)

    listo = []
    X = []

    idx = 0

    w2id = defaultdict(list)

    frequencies = []

    words = {'kind', 'mind', 'bind',
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

    english_dict.extend(dutch_dict)
    word_dict = defaultdict(list)

    for ortho, freq, phono in english_dict:

        if len(set(ortho) - set(ascii_lowercase)) > 0:
            continue

        if len(ortho) > max_len:
            print("{0} too long: {1}".format(ortho, len(ortho)))
            continue

        if ortho not in words and idx >= max_lex:
            continue

        if ortho not in words and freq < 100:
            continue

        orth = np.zeros((orth_vec_len,))
        x_orth = o.transform(ortho).ravel()

        orth[:x_orth.shape[0]] = x_orth

        try:
            x_phon = sivi.vectorize_single(phono, left=False)
        except ValueError as e:
            print("Skipped {0}".format(ortho, phono))
            print(e)
            continue

        listo.append(" ".join((ortho, phono)))
        X.append(np.hstack([orth, x_phon]))

        w2id[ortho].append(idx)
        word_dict[ortho].append(phono)

        frequencies.append(freq)
        idx += 1

    X_orig = np.array(X)
    frequencies = np.array(frequencies) + 1

    sample = np.array(random_sample(listo, frequencies, 30000, min_count=10))
    s_ = list(set(sample))
    c_ = Counter(sample)

    X = np.array([X_orig[listo.index(item)] for item in sample])
    X_orig = np.array([X_orig[listo.index(x)] for x in s_])

    start = time.time()

    # s = Lexisom((20, 20), X.shape[1], 1.0, orth_len=orth_vec_len, phon_len=X.shape[1] - orth_vec_len)
    # s.train(X, 300, total_updates=10000, batch_size=100, show_progressbar=True, stop_nb_updates=0.5)
    # s = Lexisom.load("bi_lexicon.json")
    # s = Lexisom.load("bi_freq.json")
    s = Lexisom.load("2020som_log.json")

    print("Took {0} seconds".format(time.time() - start))

    sap = defaultdict(list)
    pas = defaultdict(list)

    for k, v in zip(s_, s.predict(X_orig)):

        orth, pron = k.split()

        sap[v].append(k)
        pas[orth].append(v)
