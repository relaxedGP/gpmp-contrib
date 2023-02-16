# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------

import time
import numpy as np

def pareto_dominated(z0, z):
    """
    Get a boolean array corresponding to points in z that Pareto-dominate z0
    """
    better = (z <= z0).all(axis=-1)
    return better


def pareto_filter(z, z0):
    """
    Get a boolean array corresponding to points in z that are Pareto-dominated by z0
    """
    worse = (z >= z0).all(axis=-1)
    return worse


def pareto_points_unsorted(z):
    """
    Returns the pareto-optimal points
    :param z: nd array n_points x n_costs
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """
    is_opt = np.ones(z.shape[0], dtype=bool)

    for i, c in enumerate(z):
        if is_opt[i]:
            # Keep any point with at least a lower cost
            is_opt[is_opt] = np.any(z[is_opt] < c, axis=1)
            is_opt[i] = True  # And keep self

    return is_opt


def pareto_points(z, idx=None):
    """
    Returns the pareto-optimal points
    :param z: nd array n_points x n_costs
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """
    n_points, n_costs = z.shape

    # Initialize the boolean array to indicate which points are Pareto optimal
    is_opt = np.ones(n_points, dtype=bool)

    # Sort the points by the first cost dimension
    indices = np.argsort(z[:, 0])

    # Initialize the minimum cost array and set the first minimum to the first cost
    z = z[indices]

    for i, c in enumerate(z):
        if is_opt[i]:
            # Keep any point with at least a lower cost
            is_opt[is_opt] = np.any(z[is_opt] < c, axis=1)
            is_opt[i] = True  # And keep self

    if idx is None:
        is_opt_inv = np.empty_like(is_opt)
        is_opt_inv[indices] = is_opt
        return is_opt_inv
    else:
        idx[indices] = is_opt


def plot_pareto(axis, z_opt, color="red"):

    axis.plot(
        z_opt[:, 0], z_opt[:, 1], linestyle="", marker="o", markersize=2, color=color
    )
    ind_sort = z_opt[:, 0].argsort()
    z_opt = z_opt[ind_sort]
    n = z_opt.shape[0]
    for i in range(n - 1):
        axis.plot(
            [z_opt[i, 0], z_opt[i + 1, 0]], [z_opt[i, 1], z_opt[i, 1]], color=color
        )
        axis.plot(
            [z_opt[i + 1, 0], z_opt[i + 1, 0]],
            [z_opt[i, 1], z_opt[i + 1, 1]],
            color=color,
        )


def dominated_area_2d(z_ref, z_opt):

    n = z_opt.shape[0]
    ind_sort = z_opt[:, 0].argsort()
    z_opt = z_opt[ind_sort]
    z_0 = np.array([z_opt[0, 0], z_opt[n-1, 1]])

    B = (z_ref[0] - z_0[0]) * (z_ref[1] - z_0[1])

    if n > 1:
        width = np.diff(z_opt[:, 0])
        height = z_opt[0 : n - 1, 1] - z_0[1]
        return B - np.sum(width * height)
    else:
        return B


def symmdiff_area_2d(z_ref, z_opt_1, z_opt_2):
    z_stacked = np.vstack((z_opt_1, z_opt_2))
    b = pareto_points(z_stacked)
    z_opt_union = z_stacked[b]

    symmdiff = (
        2 * dominated_area_2d(z_ref, z_opt_union)
        - dominated_area_2d(z_ref, z_opt_1)
        - dominated_area_2d(z_ref, z_opt_2)
    )

    return symmdiff

def test_pareto():
    
    n = 1000
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    z = np.random.multivariate_normal(mean, cov, n)

    tic = time.time()
    b1 = pareto_points_unsorted(z)
    print(time.time() - tic)
    
    tic = time.time()
    b2 = pareto_points(z)
    print(time.time() - tic)

    print(np.all(b1 == b2))

    z_ref = np.max(z, axis=0)
    print(dominated_area_2d(z_ref, z))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(z[:, 0], z[:, 1])
    plot_pareto(ax, z[b2])
    plt.show()

    z1 = np.array([[0, 0.1, 0.2],[0.5, 0.2, 0]]).T
    z2 = np.array([[0, 0.2],[0.5, 0]]).T
    z_ref = np.array([1, 1])
    print(dominated_area_2d(z_ref, z1))
    print(symmdiff_area_2d(z_ref, z1, z2))
