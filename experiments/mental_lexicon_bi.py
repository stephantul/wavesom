import json
import numpy as np

from wavesom.wavesom import Wavesom, transfortho, show, normalize, softmax, sigmoid
from wordkit.construct import construct_pipeline

from collections import defaultdict


if __name__ == "__main__":

    path = "saved_models/bilingual_model_batch_1.json"

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

    construct = construct_pipeline()
    wordlist = list(wordlist)
    X, words, _, X_orig = construct.fit_transform(wordlist, return_samples=False)

    s = Wavesom.load(path, np)

    w2l = defaultdict(list)
    w2l_form = defaultdict(list)
    l2w = defaultdict(set)
    quant = {}

    for n, x, q in zip(words, s.predict(X_orig), s.quant_error(X_orig)):
        w2l_form[n.split()[0]].append(x)
        w2l[n].append(x)
        l2w[x].add(n)
        quant[n] = q

    unique_ortho = {v.split()[0]: X_orig[idx] for idx, v in enumerate(words)}

    o_words, vectors = zip(*unique_ortho.items())
    vectors = np.array(vectors)

    states = []
    deltas = []

    inv = s.invert_projection(X_orig, words)
    inv_o = s.invert_projection(vectors, o_words)

    scores = {}

    p = []
    states_arr = []

    for word, item in zip(o_words, vectors):

        x = s.activate(item[:s.orth_len], iterations=50)
        max_x = np.argmax(x[-1])
        p.append((word, max_x, w2l_form[word], inv_o[max_x]))
        states_arr.append(np.array(x).max(1))

    wrong = [x for idx, x in enumerate(p) if x[0] != x[-1]]
    score = 1 - (len(wrong) / len(p))
    states_arr = np.asarray(states_arr)

    x_z = normalize(s.distance_function(s.weights, X_orig)[0])
    # from wavesom.visualization.moviegen import moviegen
    print("WRITING")

    states = []
    s.state[:] = 1.
    states.extend(s.activate(X_orig[108, :s.orth_len], iterations=10))
    states.extend(s.activate(X_orig[46, :s.orth_len], iterations=10))

    states = np.array(states)

    from wavesom.visualization.moviegen import moviegen
    reshaped = np.array(states).reshape((len(states), 25, 25)).transpose(0, 2, 1)
    f = moviegen('drrr.gif', reshaped, l2w, write_words=True)
