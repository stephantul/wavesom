import numpy as np

from sklearn.cluster import KMeans

from wavesom import Wavesom
from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer
from wordkit.features import fourteen, binary_features
from wordkit.feature_extraction import phoneme_features
from sklearn.pipeline import FeatureUnion

if __name__ == "__main__":

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

    unique_ortho = {v.split()[0]: X[idx] for idx, v in enumerate(words)}

    o_words, vectors = zip(*unique_ortho.items())
    vectors = np.array(vectors)

    np.random.seed(44)

    num_k = 60

    k = KMeans(n_clusters=num_k)
    result = k.fit_transform(X)

    s = Wavesom((num_k, 1), X.shape[1], 1.0, 98, X.shape[1]-98)
    s.weights = k.cluster_centers_

    c = s.converge(vectors[:, :s.orth_len], batch_size=1, max_iter=10000, tol=0.0001)

    dist = np.linalg.norm(s.statify(c)[None, :, :] - X[:, None, :], axis=2).argmin(0)
    res = [(o_words[idx], words[d]) for idx, d in enumerate(dist)]
    score_reconstruction = len([x for x in res if (x[0] == x[1].split(" ")[0])]) / len(res)
