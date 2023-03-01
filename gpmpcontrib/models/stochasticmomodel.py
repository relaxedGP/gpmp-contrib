"""
Model specification for multi-objective stochastic optimization

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import jax
import numpy as np
import gpmp as gp
import jax.numpy as jnp


def kernel_ii_or_tt(x, param, output_idx, pairwise=False):
    """Covariance of the observations at points given by x
    Parameters
    ----------
    x : ndarray(n, d + output_dim)
        Data points in dimension d. The last columns contain the noise
        variance at the locations for each outputs.
    param : ndarray(1 + d)
        sigma2 and range parameters
    output_idx : where to fetch noise_variance

    """
    # parameters
    p = 2
    param_dim = param.shape[0]
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])
    d = invrho.shape[0]
    noise_idx = d + output_idx
    nugget = 10 * jnp.finfo(jnp.float64).eps

    if pairwise:
        # return a vector of covariances between predictands
        K = sigma2 * jnp.ones((x.shape[0],)) + \
            x[:, noise_idx] + nugget  # nx x 0
    else:
        # return a covariance matrix between observations
        xs = gp.kernel.scale(x[:, :d], invrho)
        K = gp.kernel.distance(xs, xs)  # nx x nx
        K = sigma2 * \
            gp.kernel.maternp_kernel(p, K) + jnp.diag(x[:, noise_idx] + nugget)

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points"""
    # parameters
    p = 2
    param_dim = param.shape[0]
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])
    d = invrho.shape[0]

    xs = gp.kernel.scale(x[:, :d], invrho)
    ys = gp.kernel.scale(y[:, :d], invrho)
    if pairwise:
        # return a vector of covariances
        K = gp.kernel.distance_pairwise(xs, ys)  # nx x 0
    else:
        # return a covariance matrix
        K = gp.kernel.distance(xs, ys)  # nx x ny

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def build_kernel(output_idx):

    def kernel(x, y, param, pairwise=False):

        if y is x or y is None:
            return kernel_ii_or_tt(x, param, output_idx, pairwise)
        else:
            return kernel_it(x, y, param, pairwise)

    return kernel


def constant_mean(x, param):
    return jnp.ones((x.shape[0], 1))


def build_anisotropic_parameters_initial_guess(output_dim):

    def anisotropic_parameters_initial_guess(model, xi, zi):

        rho = jnp.std(xi, axis=0)[:output_dim]

        covparam = jnp.concatenate((jnp.array([jnp.log(1.0)]), -jnp.log(rho)))

        # n = xi.shape[0]
        # sigma2_GLS = 1 / n * model.norm_k_sqrd(
        #     xi,
        #     zi.reshape((-1, )),
        #     covparam
        # )
        sigma2 = jnp.std(zi)

        return jnp.concatenate(
            (jnp.array([jnp.log(sigma2)]), -jnp.log(rho)))

    return anisotropic_parameters_initial_guess


def negative_log_restricted_penalized_likelihood(model,
                                                 xi,
                                                 zi,
                                                 covparam,
                                                 mean_prior,
                                                 invcov_prior):

    delta = covparam - mean_prior
    penalization = 0.5 * delta.T @ (invcov_prior @ delta)
    nlrel = model.negative_log_restricted_likelihood(xi, zi, covparam)

    return nlrel + penalization


def build_make_remap_criterion(mean_prior, invcov_prior):

    def make_remap_criterion(model, xi, zi):

        nlrepl = jax.jit(
            lambda covparam: negative_log_restricted_penalized_likelihood
            (model, xi, zi, covparam, mean_prior, invcov_prior)
        )
        dnlrepl = jax.grad(nlrepl)

        return nlrepl, dnlrepl

    return make_remap_criterion


def build_models(output_dim):

    models = [{
        "name": "",
        "model": None,
        "parameters_initial_guess_procedure": None,
        "make_selection_criterion": None,
    } for i in range(output_dim)]

    # same hyper prior for all outputs

    # TODO : set mean prior for sigma^2 adaptively
    mean_prior = np.array([0, -np.log(1 / 3), -np.log(1 / 3)])
    invcov_prior = np.diag([0, 1 / np.log(10/3)**2, 1 / np.log(10/3)**2])

    # same remap criterion for all outputs
    make_remap_criterion = build_make_remap_criterion(mean_prior, invcov_prior)

    for i in range(output_dim):
        covariance = build_kernel(i)

        models[i]["model"] = gp.core.Model(constant_mean,
                                           covariance,
                                           None,
                                           None)
        models[i][
            "parameters_initial_guess_procedure"
        ] = build_anisotropic_parameters_initial_guess(output_dim)
        
        models[i]["make_selection_criterion"] = make_remap_criterion

    return models
