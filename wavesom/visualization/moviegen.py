import matplotlib
matplotlib.use('Agg')

import time
import numpy as np
import matplotlib.pyplot as plt

from wavesom.visualization.images2gif import writeGif
from wavesom.experiments.mental_lexicon_bi import transfortho
from PIL import Image


def moviegen(word, fn, to, num_neighbors, s, gen_func, sap, o, orth_vec_len, start_idx=0, end_idx=None):

    if end_idx is None:
        end_idx = orth_vec_len

    a = transfortho(word, o, orth_vec_len)

    frames = []

    start = time.time()
    res = s.activate_values(a, max_depth=to, num=num_neighbors, start_idx=start_idx, end_idx=end_idx)
    print("Took {0} seconds".format(time.time() - start))

    for x in range(1, to):

        b = np.array(list(res.values())[:x]).mean(axis=0)

        f = plt.figure(figsize=(12, 12), dpi=80)
        gen_func(sap, s.map_dimensions[0], b)
        f.canvas.draw()
        data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(np.copy(data)))
        plt.close()

    writeGif(fn, images=frames, duration=0.2)

