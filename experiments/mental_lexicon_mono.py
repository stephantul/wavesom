import numpy as np
import json

from wavesom.wavesom import Wavesom
from experiments.setup import setup


if __name__ == "__main__":

    s = Wavesom.load("saved_models/monolingual_model.json")
    # c = json.load(open("monolingual_model.json_counts.json"))

    dicto = json.load(open("data/syllable_lexicon.json"))

    dicto = [k for k in dicto if k[2] != 'dut']

    max_len = 7
    X, X_orig, s_ = setup(dicto, max_len, [], 100000)

    orth_vec_len = 14 * max_len

    err = s.quant_error(X)
