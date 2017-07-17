import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from .images2gif import writeGif

import time
import numpy as np
import matplotlib.pyplot as plt
import visdom

from PIL import Image

v = np.array([[0.267004,  0.004874,  0.329415],
              [0.270595,  0.214069,  0.507052],
              [0.19943,  0.387607,  0.554642],
              [0.13777,  0.537492,  0.554906],
              [0.157851,  0.683765,  0.501686],
              [0.440137,  0.811138,  0.340967]])


l = ListedColormap(v)


def moviegen(fn, activations, l2w, *, write_words=False):

    frames = []

    size = activations[0].shape[0]

    max_value = activations.max()

    for idx, x in enumerate(activations):

        f = plt.figure(figsize=(5, 5), dpi=80)
        plt.imshow(x, extent=[0, x.shape[0], x.shape[1], 0], interpolation=None, vmin=0.0, vmax=1.0)
        plt.annotate(idx, (1, 1), color='white')
        if write_words:
            for idx, words in l2w.items():
                plt.annotate("\n".join(words), ((idx // size)+.5, (idx % size)+.5), color='white', size=10)
        f.canvas.draw()
        data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(np.copy(data)))
        plt.close()

    start = time.time()
    print("Write")
    frames[0].save(fn, save_all=True, append_images=frames[1:])
    print("STOPe: {}".format(time.time() - start))
