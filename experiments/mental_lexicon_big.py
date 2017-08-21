import json
import numpy as np

from wavesom.wavesom import Wavesom
from wordkit.construct import construct_pipeline

from collections import defaultdict


if __name__ == "__main__":

    path = "saved_models/biling_lots_of_words.json"

    wordlist = json.load(open("data/words.json"))

    '''wordlist = {'kind', 'mind', 'bind',
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
                'creek', 'ziek', 'piek'}'''

    construct = construct_pipeline(corpus_types=(), languages=(), corpus_paths=())
    wordlist = list(wordlist)
    (X_orig, words), () = construct.fit_transform(wordlist)

    words = [" ".join((x["orthography"], "-".join(x["syllables"]))) for x in words]

    s = Wavesom.load(path, np)
    s.orth_len = construct.stages[-1][1].transformer_list[0][1].vec_len

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

    for idx, (word, item) in enumerate(zip(words, X_orig)):

        print(idx)

        s.state[:] = 1.
        x = s.activate(item, iterations=1000)
        max_x = np.argmax(x[-1])
        p.append((word.split()[0], max_x, w2l_form[word], inv_o[max_x]))
        states_arr.append(x[-1])

    wrong = [x for idx, x in enumerate(p) if x[0] != x[-1] and x[1] not in x[2]]
    score = 1 - (len(wrong) / len(p))
    states_arr = np.asarray(states_arr)

    states = []
    s.state[:] = 1.

    states.append(s.activate(X_orig[-2, :s.orth_len], iterations=1000))
    states.append(s.activate(X_orig[49, :s.orth_len], iterations=1000))
    states.append(s.activate(X_orig[-2, :s.orth_len], iterations=1000))

    states = np.array(states)
    states = states[~(np.arange(len(states)) % 100).astype(bool)]
    # print(len(states))

    from sklearn.decomposition import PCA
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from matplotlib import pyplot as plt
    p = PCA(n_components=2, whiten=False)
    states_transformed = p.fit_transform(states_arr)

    f = plt.figure()
    v = Voronoi(states_transformed)
    voronoi_plot_2d(v, show_vertices=False, show_points=False)
    for idx, (x, y) in enumerate(states_transformed):
        plt.text(x, y, words[idx], fontsize=5)

    colors = ['red', 'green', 'blue']

    for idx, a in enumerate(states):
        trajectory = p.transform(a)
        ttt = np.hstack([trajectory[1:], trajectory[:-1]])
        if idx != 0:
            plt.plot((x1, trajectory[0][0]), (y1, trajectory[0][1]), colors[idx % len(colors)])

        for x0, y0, x1, y1 in ttt:
            plt.plot((x0, x1), (y0, y1), colors[idx % len(colors)])

    #from wavesom.visualization.moviegen import moviegen
    #reshaped = np.array(states).reshape((len(states), 10, 10)).transpose(0, 2, 1)
    #f = moviegen('drrr.gif', reshaped, l2w, write_words=False)
