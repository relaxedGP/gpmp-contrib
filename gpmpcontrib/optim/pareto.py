import numpy as np
import matplotlib.pyplot as plt


def pareto_filter(z, z0):
    """
    Get a boolean array corresponding to points in z that are Pareto-dominated by z0
    """
    worse = (z >= z0).all(axis=-1)
    return worse


def pareto_points(z):
    """
    Returns the pareto-optimal points
    :param z: nd array n_points x n_costs
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """
    is_opt = np.ones(z.shape[0], dtype = bool)
    
    for i, c in enumerate(z):
        if is_opt[i]:
            is_opt[is_opt] = np.any(z[is_opt] < c, axis=1)  # Keep any point with at least a lower cost
            is_opt[i] = True  # And keep self
    
    return is_opt

def plot_pareto(z_opt, color='red'):
    plt.plot(z_opt[:, 0], z_opt[:, 1], linestyle='', marker='o', markersize=2, color=color)
    ind_sort = z_opt[:, 0].argsort()
    z_opt = z_opt[ind_sort]
    n = z_opt.shape[0]
    for i in range(n-1):
        plt.plot([z_opt[i, 0], z_opt[i+1, 0]], [z_opt[i, 1], z_opt[i, 1]], color=color)
        plt.plot([z_opt[i+1, 0], z_opt[i+1, 0]], [z_opt[i, 1], z_opt[i+1, 1]], color=color)
