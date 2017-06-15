import json
import numpy as np

from wavesom.wavesom import Wavesom, transfortho, show
from experiments.setup import setup
from experiments.preprocessing.orthographizer.mcclelland import mcclelland_orthography

from collections import defaultdict

path = "saved_models/dim50400epochs.json"
dicto = json.load(open("data/syllable_lexicon.json"))

max_len = 7

np.random.seed(44)

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

o = mcclelland_orthography()

X, X_orig, unique_words, counts = setup(dicto, max_len, wordlist)
orth_vec_len = len(o.data["A"]) * max_len

s = Wavesom.load(path, np)

w2l = defaultdict(list)
l2w = defaultdict(list)

for n, x in zip(unique_words, s.predict(X_orig)):
    w2l[n].append(x)
    l2w[x].append(n)

# a, b = [y for x, y in w2l.items() if x.startswith("keel")]
# w1 = s.weights[a, :]
# w2 = s.weights[b, :]

# p = transfortho("KEEL", o, s.weights.shape[1])
# dist1 = np.linalg.norm(p[None, None, :] - w1[None, :, :])
# dist2 = np.linalg.norm(p[None, None, :] - s.weights[:, None, :], axis=2)
import cProfile
cProfile.run("x = s.activate_state(X_orig[1], 8, 12)")
print("Done, dome")

# inv = {idx: k for idx, k in enumerate(s.invert_projection(X_orig, unique_words))}
# from wavesom.visualization.simple_viz import show_labelmap
# show_labelmap(inv, 50)
