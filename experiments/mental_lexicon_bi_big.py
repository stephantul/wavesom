import numpy as np
import json

from wavesom.wavesom import Wavesom
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer
from wordkit.features import fourteen, binary_features
from wordkit.feature_extraction import phoneme_features
from sklearn.pipeline import FeatureUnion

from collections import defaultdict


if __name__ == "__main__":

    path = "saved_models/test_model.json"

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

    wordlist = json.load(open("data/shared_vocab.json"))
    wordlist = {x.lower() for x in wordlist if "-" not in x}

    o = LinearTransformer(fourteen)
    p = ONCTransformer(phoneme_features(binary_features))

    transformers = FeatureUnion([("o", o), ("p", p)])

    c_d = Celex("data/dpl.cd", language='nld', merge_duplicates=True)
    c_e = Celex("data/epl.cd", language='eng', merge_duplicates=True)

    corpora = FeatureUnion([("d", c_d), ("e", c_e)])

    words = corpora.fit_transform(wordlist)
    X = transformers.fit_transform(words)
    words = [" ".join((x['orthography'], "-".join(x['syllables']))) for x in words]

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

    states = []
    deltas = []

    inv = s.invert_projection(X, words)
    inv_o = s.invert_projection(vectors, o_words).reshape(s.weight_dim)

    scores = {}

    results = []
    states_arr = []

    for idx, (word, item) in enumerate(zip(words, X)):

        print("{} / {}".format(idx, len(words)))

        s.state[:] = 1.
        x = s.converge(item, max_iter=10000)
        max_x = np.argmax(x[-1])
        results.append((word.split()[0], max_x, w2l_form[word.split()[0]], inv_o[max_x], len(x)))
        states_arr.append(x[-1])

    wrong = [x for idx, x in enumerate(results) if x[0] != x[-2] and x[1] not in x[2]]
    score = 1 - (len(wrong) / len(results))
    states_arr = np.asarray(states_arr)

    np.save(open("data/states_arr_big.npy", 'wb'), states_arr)

    states_arr_o = []

    for idx, (word, item) in enumerate(zip(o_words, vectors)):

        print("OO: {} / {}".format(idx, len(words)))

        s.state[:] = 1.
        x = s.converge(item[:o.vec_len], max_iter=10000)
        max_x = np.argmax(x[-1])
        states_arr_o.append(x[-1])

    states_arr_o = np.asarray(states_arr_o)

    np.save(open("data/states_arr_o_big.npy", 'wb'), states_arr_o)

    # states_arr = np.load(open("data/states_arr_amlap.npy", 'rb'))
    # states_arr_o = np.load(open("data/states_arr_o_amlap.npy", 'rb'))

    states = []
    s.state[:] = 1.

    # word_to_converge = np.random.choice(o_words, size=5, replace=True)
    word_to_converge = [o_words[o_words.index("kind")],
                        o_words[o_words.index("wind")],
                        o_words[o_words.index("dine")]]

    s.activate(X[65, :s.orth_len], iterations=1000)

    for word in word_to_converge:
        print(word)
        states.append(s.converge(X[w2id[word][0], :o.vec_len], max_iter=10000))

    states = np.array(states)
    # states = states[~(np.arange(len(states)) % 100).astype(bool)]
    # print(len(states))

    from sklearn.decomposition import PCA
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from matplotlib import pyplot as plt

    p = PCA(n_components=2)
    states_transformed = p.fit_transform(states_arr)

    v = Voronoi(states_transformed)
    voronoi_plot_2d(v, show_vertices=False, show_points=False)
    for idx, (x, y) in enumerate(states_transformed):
        if words[idx].split()[0] in word_to_converge:
            plt.text(x, y, words[idx], fontsize=10)

    '''colors = [(((.5 / (len(words) - 1)) * x)) for x in range(len(words))]
    colors = np.vstack([colors, colors, colors]) + .5
    colors = (colors.T * 255).round(0).astype(int)
    colors = ["#{0}{1}{2}".format(hex(x)[-2:], hex(y)[-2:], hex(z)[-2:]) for x, y, z in colors'''

    colors = ['teal', 'green', 'orange', 'purple']

    for idx, a in enumerate(states):
        trajectory = p.transform(a)
        ttt = np.hstack([trajectory[1:], trajectory[:-1]])
        if idx != 0:
            plt.plot((x1, trajectory[0][0]), (y1, trajectory[0][1]), colors[idx % len(colors)])

        for x0, y0, x1, y1 in ttt:
            plt.plot((x0, x1), (y0, y1), colors[idx % len(colors)])

    plt.axis('off')
    # plt.savefig("temp.png")
    # plt.close()

    # states = []

    # for idx, a in enumerate(vectors):
    #     states.append(s.converge(a[: o.vec_len], max_iter=10000)[-1])

    '''# PICTURE FOR AMLAP STUFF
    import matplotlib as mpl

    teal = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['white', 'teal'], 256)
    teal._init()
    gree = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['white', 'green'], 256)
    gree._init()
    yell = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['white', 'orange'], 256)
    yell._init()
    pink = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', ['white', 'purple'], 256)
    pink._init()

    alphas = np.linspace(0, 1.0, teal.N+3)
    teal._lut[:, -1] = alphas
    gree._lut[:, -1] = alphas
    yell._lut[:, -1] = alphas
    pink._lut[:, -1] = alphas

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(states_arr[69].reshape(25, 25), cmap=teal, interpolation=None)
    plt.axis('off')
    fig.savefig("test_fig_reign.png", bbox_inches='tight', transparent=True)

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(states_arr[89].reshape(25, 25), cmap=gree, interpolation=None)
    plt.axis('off')
    fig.savefig("test_fig_feign.png", bbox_inches='tight', transparent=True)

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(states_arr[103].reshape(25, 25), cmap=yell, interpolation=None)
    plt.axis('off')
    fig.savefig("test_fig_wind.png", bbox_inches='tight', transparent=True)

    fig = plt.figure(figsize=(50, 50))
    plt.imshow(states_arr[104].reshape(25, 25), cmap=pink, interpolation=None)
    plt.axis('off')
    fig.savefig("test_fig_waind.png", bbox_inches='tight', transparent=True)

    #from wavesom.visualization.moviegen import moviegen
    #reshaped = np.array(states).reshape((len(states), 10, 10)).transpose(0, 2, 1)
    #f = moviegen('drrr.gif', reshaped, l2w, write_words=False)'''
