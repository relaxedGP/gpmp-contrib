# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
#          Sébastien Petit
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
from scipy.stats import norm
import time
import numpy as np
import gpmp.num as gnp
import gpmp as gp
import sklearn.neighbors
from .smc import ParticlesSet
from copy import deepcopy


def get_one_nn_predictions(x, y, box, rng, N=100000):
    grid = ParticlesSet.randunif(x.shape[1], N, box, rng).numpy()

    knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1)
    knn.fit(x, y)
    knn_pred = knn.predict(grid)

    return knn_pred

def get_rectified_spatial_quantile(xi, zi, box, rng, l):
    nn_predictions = get_one_nn_predictions(xi, zi, box, rng)
    spatial_quantile = np.quantile(nn_predictions, l)
    raw_quantile = np.quantile(zi, l)

    return max(spatial_quantile, raw_quantile)

def one_sided(t0, min_value, max_value, n_ranges):
    assert t0 > min_value, (min_value, max_value)

    G = [- np.inf, float(t0)]

    t = np.logspace(np.log10(t0 - min_value), np.log10(max_value - min_value), n_ranges) + min_value
    t[-1] = np.inf
    R_list = [[[float(_t), np.inf]] for _t in t]

    return G, R_list

optim_strategy = {
    "Constant": lambda l, rng, box, options: lambda xi, zi: one_sided(
        np.quantile(zi[:options["n_init"]], l), zi.min(), zi.max(), options["n_ranges"]
    ),
    "Concentration": lambda l, rng, box, options: lambda xi, zi: one_sided(
        np.quantile(zi, l), zi.min(), zi.max(), options["n_ranges"]
    ),
    "Spatial": lambda l, rng, box, options: lambda xi, zi: one_sided(
        get_rectified_spatial_quantile(xi, zi, box, rng, l), zi.min(), zi.max(), options["n_ranges"]
    ),
}

def two_sided(t, d, xi, zi, n_ranges):
    assert d > 0, (t, d, zi)

    G = [float(t - d), float(t + d)]

    excursion_range = max(
        zi.max() - G[1],
        G[0] - zi.min(),
    )

    side_spread = np.logspace(
        np.log10(d),
        np.log10(d + excursion_range),
        num=n_ranges
    )

    R_list = [[[-np.inf, float(t - _s)], [float(t + _s), np.inf]] for _s in side_spread]
    R_list[-1] = [[-np.inf, -np.inf], [np.inf, np.inf]]

    return G, R_list


levelset_strategy = {
    "Constant": lambda l, rng, box, options: lambda xi, zi: two_sided(
        options["t"], np.quantile(np.abs(zi[:options["n_init"]] - options["t"]), l), xi, zi, options["n_ranges"]
    ),
    "Concentration": lambda l, rng, box, options: lambda xi, zi: two_sided(
        options["t"], np.quantile(np.abs(zi - options["t"]), l), xi, zi, options["n_ranges"]
    ),
    "Spatial": lambda l, rng, box, options: lambda xi, zi: two_sided(
        options["t"], get_rectified_spatial_quantile(xi, np.abs(zi - options["t"]), box, rng, l), xi, zi, options["n_ranges"]
    ),
}

def get_membership_indices(zi, R):
    """
    Get the index of the membership of each row in zi to an interval in R.

    Parameters
    ----------
    zi : ndarray
        An (n, 1) shaped array of data points.
    R : list of lists
        A list of intervals represented as [l_i, u_i] pairs.

    Returns
    -------
    ei : ndarray
        An (n, 1) shaped array of membership indices.
    """

    n = zi.shape[0]
    ei = np.zeros(n, dtype=int)

    for idx, (l, u) in enumerate(R, start=1):
        mask = np.logical_and(zi > l, zi <= u)
        c = idx * mask
        ei = ei + c

    return ei


def split_data(xi, zi, ei, R):
    """Split the data based on the membership indices.

    Parameters
    ----------
    xi : ndarray
        An (n, d) shaped array of data points.
    zi : ndarray
        An (n, 1) shaped array of data points.
    ei : ndarray
        An (n, 1) shaped array of membership indices.
    R : list of intervals
        List of intervals, each specified as [l_k, u_k].

    Returns
    -------
    (x0, z0, ind0) : tuple of ndarrays
        A tuple of (n0, d), (n0, 1) shaped arrays for xi and zi rows
        with ei = 0, and the corresponding indices.
    (x1, z1, bounds, ind1) : tuple of ndarrays
        Rows of xi and zi with ei > 0, and the corresponding interval
        bounds and indices.

    """

    mask = ei.reshape(-1) == 0
    ind0 = np.where(mask)[0]
    ind1 = np.where(~mask)[0]

    x0, z0 = xi[mask], zi[mask]

    x1, z1, e1 = xi[~mask], zi[~mask], ei[~mask]

    interval_indices = e1 - 1
    bounds = [R[i] for i in interval_indices]

    return (x0, z0, ind0), (x1, z1, bounds, ind1)


def make_regp_criterion_with_gradient(model, x0, z0, x1, meanparam_dim):
    """
    Make regp criterion function with gradient.

    Parameters
    ----------
    model : gpmp model
        Gaussian process model.
    x0 : ndarray, shape (n0, d)
        Locations of the observed data points not relaxed
    z0 : ndarray, shape (n0,)
        Observed values at the data points not relaxed
    x1 : ndarray, shape (n1, d)
        Locations of the relaxed  data points
    meanparam_dim : int,
        Number of dimension of the mean parameter

    Returns
    -------
    crit_jit : function
        Selection criterion function with gradient.
    dcrit : function
        Gradient of the selection criterion function.
    """
    x0 = gnp.asarray(x0)
    x1 = gnp.asarray(x1)
    z0 = gnp.asarray(z0)

    xi = gnp.vstack((x0, x1))

    n1 = x1.shape[0]

    # selection criterion

    selection_criterion = model.negative_log_likelihood

    def crit_(param):
        meanparam = param[0:meanparam_dim]

        param = param[meanparam_dim:]

        if n1 > 0:
            covparam = param[0:-n1]
            z1 = param[-n1:]
        elif n1 == 0:
            covparam = param
            z1 = gnp.array([])
        else:
            raise ValueError(n1)

        zi = gnp.concatenate((z0, z1))
        l = selection_criterion(meanparam, covparam, xi, zi)
        return l

    crit_jit = gnp.jax.jit(crit_)

    dcrit = gnp.jax.jit(gnp.grad(crit_jit))

    return crit_jit, dcrit


def remodel(
        model, xi, zi, R, covparam_bounds, info=False, verbosity=0, optim_options={},
):
    """
    Perform reGP optimization (REML + relaxation)

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    R : list of intervals
        List of relaxation intervals, each specified as [l_k, u_k].
    info : bool, optional
        Whether to return additional information.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    model : GPmp model
        Updated GPmp Gaussian process model.
    zi_relaxed : ndarray, shape (n,)
        Relaxed output data.
    ind_relaxed : ndarray
        Indices of the relaxed data points in the input zi array.
    info_ret : dict, optional
        Additional information (if info=True).
    """
    if optim_options["relaxed_init"] in ["flat", "f-values"]:
        return _remodel(model, xi, zi, R, covparam_bounds, info=info, verbosity=verbosity, optim_options=optim_options)
    else:
        assert optim_options["relaxed_init"] == "both"

    _optim_options = deepcopy(optim_options)

    _optim_options["relaxed_init"] = "flat"
    nll_1 = _remodel(model, xi, zi, R, covparam_bounds, info=True, verbosity=verbosity, optim_options=_optim_options)[3].fun
    _optim_options["relaxed_init"] = "f-values"
    nll_2 = _remodel(model, xi, zi, R, covparam_bounds, info=True, verbosity=verbosity, optim_options=_optim_options)[3].fun

    print("NLL1 : {}, NLL2: {}".format(nll_1, nll_2))
    if nll_1 < nll_2:
        _optim_options["relaxed_init"] = "flat"
    return _remodel(model, xi, zi, R, covparam_bounds, info=info, verbosity=verbosity, optim_options=_optim_options)

def _remodel(
        model, xi, zi, R, covparam_bounds, info=False, verbosity=0, optim_options={},
):
    """
    Perform reGP optimization (REML + relaxation)

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    R : list of intervals
        List of relaxation intervals, each specified as [l_k, u_k].
    info : bool, optional
        Whether to return additional information.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    model : GPmp model
        Updated GPmp Gaussian process model.
    zi_relaxed : ndarray, shape (n,)
        Relaxed output data.
    ind_relaxed : ndarray
        Indices of the relaxed data points in the input zi array.
    info_ret : dict, optional
        Additional information (if info=True).
    """
    tic = time.time()

    # Membership indices and split data
    ei = get_membership_indices(gnp.to_np(zi), R)
    (x0, z0, ind0), (x1, z1, z1_bounds, ind1) = split_data(xi, gnp.to_np(zi), ei, R)
    z1_size = z1.shape[0]

    if optim_options['relaxed_init'] == 'flat':
        z1_relaxed_init = np.zeros(z1.shape)
    
        for i in range(z1_relaxed_init.shape[0]):
            Ri = z1_bounds[i]
            if Ri[0] > -np.inf and Ri[1] < np.inf:
                z1_relaxed_init[i] = (Ri[0] + Ri[1])/2
            elif Ri[0] > -np.inf and Ri[1] == np.inf:
                if len(R) == 1:
                    z0_min = z0.min()
                else:
                    z0_min = min([_r[1] for _r in R if _r[1] < Ri[0]])

                z1_relaxed_init[i] = Ri[0] + 2 * (Ri[0] - z0_min)
            elif Ri[0] == -np.inf and Ri[1] < np.inf:
                if len(R) == 1:
                    z0_max = z0.max()
                else:
                    z0_max = max([_r[0] for _r in R if _r[0] > Ri[1]])

                z1_relaxed_init[i] = Ri[1] + 2 * (Ri[1] - z0_max)
            else:
                raise RuntimeError
    elif optim_options['relaxed_init'] == 'f-values':
        z1_relaxed_init = z1
    else:
        raise ValueError(
            'Non-supported init option for relaxed observations: {}.'.format(optim_options['relaxed_init'])
        )

    # Initial guess for the parameters
    meanparam0, covparam0 = gp.kernel.anisotropic_parameters_initial_guess_constant_mean(
        model,
        np.vstack((x0, x1)),
        np.concatenate((z0, z1_relaxed_init))
    )

    meanparam0 = meanparam0.reshape(1)

    covparam_dim = covparam0.shape[0]

    meanparam_dim = meanparam0.shape[0]
    meanparam_bounds = [(-gnp.inf, gnp.inf)] * meanparam_dim

    # Initial parameter vector and bounds
    p0 = np.concatenate((meanparam0.reshape(1), covparam0, z1_relaxed_init))

    bounds = meanparam_bounds + covparam_bounds + [tuple(_z1_bounds) for _z1_bounds in z1_bounds]

    # reGP criterion
    nlrl, dnlrl = make_regp_criterion_with_gradient(model, x0, z0, x1, meanparam_dim)

    # Verbosity level
    silent = True
    if verbosity == 1:
        print("reGP stuff...")
    elif verbosity == 2:
        silent = False

    # Check if p0 is admissible
    assert all([(bounds[i][0] <= p0[i]) and (p0[i] <= bounds[i][1]) for i in range(p0.shape[0])]), (p0, bounds)

    # Optimize parameters
    _opts = {k: v for (k, v) in optim_options.items() if k not in ['method', 'relaxed_init']}
    popt, info_ret = gp.kernel.autoselect_parameters(
        p0, nlrl, dnlrl, bounds=bounds, silent=silent, info=True, method=optim_options['method'], method_options=_opts
    )

    assert not np.isnan(popt).any()

    if verbosity == 1:
        print("done.")

    # Update the model and relaxed data
    model.meanparam = gnp.asarray(popt[0:meanparam_dim])

    model.covparam = gnp.asarray(popt[meanparam_dim:(covparam_dim + meanparam_dim)])

    z1_relaxed = popt[(covparam_dim+meanparam_dim):]

    zi_relaxed = gnp.zeros(zi.shape)
    if gnp._gpmp_backend_ == 'jax':
        zi_relaxed = zi_relaxed.at[ind0].set(gnp.asarray(z0))
        zi_relaxed = zi_relaxed.at[ind1].set(gnp.asarray(z1_relaxed))
    else:
        zi_relaxed[ind0] = gnp.asarray(z0)
        zi_relaxed[ind1] = gnp.asarray(z1_relaxed)

    # Return results
    if info:
        info_ret["covparam0"] = covparam0
        info_ret["covparam"] = model.covparam
        info_ret["meanparam0"] = meanparam0
        info_ret["meanparam"] = model.meanparam
        info_ret["selection_criterion"] = nlrl
        info_ret["time"] = time.time() - tic
        return model, zi_relaxed, ind1, info_ret
    else:
        return model, zi_relaxed, ind1


def predict(model, xi, zi, xt, R, covparam0=None, info=False, verbosity=0):
    """
    Perform reGP optimization (REML + relaxation) and prediction

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    xt : ndarray, shape (n, d)
        Locations of points to be predicted.
    R : list of intervals
        List of relaxation intervals, each specified as [l_k, u_k].
    covparam0 : ndarray, optional
        Initial guess for the covariance parameters.
    info : bool, optional
        Whether to return additional information.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    zi_relaxed : ndarray, shape (n,)
        Relaxed output data.
    (zpm, zpv) : tuple of ndarrays
        Relaxed posterior mean and variance.
    model : GPmp model
        Updated GPmp Gaussian process model.
    info_ret : dict, optional
        Additional information (if info=True).
    """
    if info is True:
        model, zi_relaxed, ind_relaxed, info_ret = remodel(
            model, xi, zi, R,
            covparam0=None,
            info=info,
            verbosity=verbosity
        )
    else:
        model, zi_relaxed, ind_relaxed = remodel(
            model, xi, zi, R,
            covparam0=None,
            info=info,
            verbosity=verbosity
        )
        info_ret = None

    zpm, zpv = model.predict(xi, zi_relaxed, xt)

    return zi_relaxed, (zpm, zpv), model, info_ret

def select_optimal_R(model, xi, zi, G, R_list, covparam_bounds, optim_options):
    """
    Choose threshold for reGP with relaxation above t0

    This function selects an optimal threshold for a reGP above t0 by
    minimizing the truncated continuous ranked probability score
    (tCRPS) over a range of possible thresholds.

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    G : interval (specified as [l, u])
        Validation range.
    R_list : list of lists of intervals (specified as [l, u])
        Relaxation range candidates.
    covparam_bounds : ndarray
        Bounds for covariance parameters.
    optim_options : dict
        Options passed to remodel

    Returns
    -------
    Rgopt : list of intervals (specified as [l, u])
        Optimal relaxation range.
    """
    q = len(R_list)

    J = gnp.numpy.zeros(q)
    for i in range(q):
        model, zi_relaxed, _ = remodel(model, xi, zi, R_list[i], covparam_bounds, optim_options=optim_options)
        zloom, zloov, _ = model.loo(xi, zi_relaxed)
        tCRPS = gp.misc.scoringrules.tcrps_gaussian(zloom, gnp.sqrt(zloov), zi_relaxed, a=G[0], b=G[1])
        J[i] = gnp.sum(tCRPS)

    iopt = gnp.argmin(gnp.asarray(J))
    Ropt = R_list[iopt]

    return Ropt


# ---------------------------------------
