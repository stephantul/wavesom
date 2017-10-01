import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


def plot_energy_surface(state, map_dimensions):

    state = state.reshape(map_dimensions)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y = np.meshgrid(*[np.arange(x) for x in map_dimensions])

    ax.plot_surface(x, y, state, antialiased=False)
    plt.show()
