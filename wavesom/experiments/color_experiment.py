import numpy as np
import cProfile

from somplay.som import Som
from somplay.batch.som import Som as BSom
from somplay.utils import MultiPlexer, linear, expo

X = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

colors = []

for x in range(10):
    for y in range(10):
        for z in range(10):
            colors.append((x / 10, y / 10, z / 10))

X = np.array(colors)

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# initialize
# bs = BSom((10, 10), dim=3, learning_rate=1.0)
s = Som((10, 10), dim=3, learning_rate=1.0)
bs = BSom((10, 10), dim=3, learning_rate=1.0)

# train
mapped_init = np.copy(s.map_weights())

cProfile.run("s.train(X, 100, total_updates=1000, show_progressbar=False)")
cProfile.run("bs.train(X, 100, total_updates=1000, batch_size=1000, show_progressbar=False)")

# predict: get the index of each best matching unit.
predictions = s.predict(X)
# quantization error: how well do the best matching units fit?
quantization_error = s.quant_error(X)
# inversion: associate each node with the x that fits best.
# inverted = s.invert_projection(X, color_names)
# Mapping: get weights, mapped to the grid points of the SOM
mapped = s.map_weights()
print("done")

# import matplotlib.pyplot as plt

# plt.imshow(mapped)
