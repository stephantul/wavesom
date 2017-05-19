from itertools import chain

import numpy as np
from somplay.experiments.preprocessing.corpus import read_carmel
from somplay.experiments.preprocessing.sample_from_corpus import sample_sentences_from_corpus, create_orth_phon_dictionary
from somplay.experiments.preprocessing.ortho import Orthographizer
from somplay.experiments.preprocessing.sivi.ipapy_features import ipapy_sivi


if __name__ == "__main__":

    english = read_carmel("data/dpc_en_carmel.txt", "data/dpc_en_carmelled.txt")
    dutch = read_carmel("data/dpc_nl_carmel.txt", "data/dpc_nl_carmelled.txt")

    max_o = 10
    max_phon = 3

    phones = set()
    for x in chain(english.values(), dutch.values()):
        phones.update(x)

    o = Orthographizer()
    p = ipapy_sivi(3)

    english = create_orth_phon_dictionary(english, orthography=o, phonology=p)
    dutch = create_orth_phon_dictionary(dutch, orthography=o, phonology=p)

    Xeng, Yeng, weng = sample_sentences_from_corpus(["data/dpc_en.txt"],
                                                    english,
                                                    30000)

    print("Loaded English")

    Xdut, Ydut, wdut = sample_sentences_from_corpus(["data/dpc_nl.txt"],
                                                    dutch,
                                                    30000)

    print("Loaded Dutch")

    X = np.concatenate([Xeng, Xdut])
    Y = np.concatenate([Yeng, Ydut])

    #s_phon = Som(100, 100, X.shape[1], 0.3)
    #s_orth = Som(100, 100, Y.shape[1], 0.3)

    #h = Hebbian(s_phon, s_orth, 0.3)
    #h.run_samples(X, Y, samples_per_epoch=100000, batch_size=10000, num_epochs=10)
