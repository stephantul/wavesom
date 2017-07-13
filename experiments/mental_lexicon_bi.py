import json
import numpy as np

from wavesom.wavesom import Wavesom, transfortho, show, normalize, softmax, sigmoid, weird_normalize
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

    states = []
    deltas = []

    '''p = []
    states_arr = []

    for word, item in zip(s_, X_orig):

        word = n = " ".join([word[0], "-".join(word[1])])
        states = []

        for idx in range(50):
            x, y = s.activate(item[:orth_vec_len])
            states.append(x)

        p.append((word, states[-1].argmax(), w2l[word]))
        states_arr.append(states[-1])

    p = sorted(p, key=lambda x: x[0].split()[0])
    states_arr = np.array(states_arr)'''

    for idx in range(1):
        x, y = s.activate(X_orig[0, :orth_vec_len])
        states.append(x)

    for idx in range(150):

        x, y = s.activate()
        states.append(x)

    x_z = sigmoid(s.distance_function(X_orig, s.weights)[0])

    from wavesom.visualization.moviegen import moviegen
    print("WRITING")
    # states = np.array(states)
    # deltas = np.array(deltas)
    # p = np.array([normalize((s.cache * x).sum(0)) for x in states])

    import cProfile
    moviegen('drrr.gif', np.array(states).reshape((len(states), 25, 25)).transpose(0, 2, 1), l2w, write_words=False)

    # TODO: Thresholding: count only neurons with activations above threshold? Maybe?
    # TODO: no?
