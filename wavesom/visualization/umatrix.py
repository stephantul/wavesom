from matplotlib import pyplot as plt
from pylab import contour
import numpy as np

from scipy.spatial import distance_matrix


def calculate_map_dist(width, height):
    """
    Calculates the grid distance, which will be used during the training
    steps. It supports only planar grids for the moment
    """

    nnodes = width * height
    d_mtr = np.zeros((nnodes, nnodes))

    for idx in range(nnodes):
        d_mtr[idx] = calculate_map_dists(width, height, idx).reshape(1, nnodes)

    return d_mtr


def calculate_map_dists(width, height, index):

    r = np.arange(0, width, 1)[:, np.newaxis]
    c = np.arange(0, height, 1)
    dist2 = (r-(index // height))**2 + (c-(index % height))**2
    dist = dist2.ravel()

    return dist


def build_u_matrix(weights, width, height, distance=1):

    ud2 = calculate_map_dist(width, height)
    u_matrix = np.zeros((weights.shape[0], 1))

    for idx, vec in enumerate(weights):
        vec = vec[np.newaxis, :]
        neighborbor_ind = ud2[idx, :] <= distance
        neighbors = weights[neighborbor_ind]
        u_matrix[idx] = distance_matrix(
            vec, neighbors).mean()

    return u_matrix.reshape((width, height))


def create_u_matrix(weights, width, sap, distance2=1):

    umat = build_u_matrix(weights, width, width, distance=distance2)
    plt.axis([0, width, 0, width])

    mn = np.min(umat.flatten())
    mx = np.median(umat.flatten())

    sp = np.linspace(mn, mx, 10)

    contour(umat, sp, linewidths=0.7,
            cmap=plt.cm.get_cmap('Greens'))

    for k, v in sap.items():
        plt.annotate("\n".join(v), ((k // width) + 0.5, (k % width) + 0.5), size=5)

    plt.show()
