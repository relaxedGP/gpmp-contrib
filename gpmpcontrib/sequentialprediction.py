"""Helper functions for sequential prediction

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import numpy as np
import gpmp as gp


class SequentialPrediction:
    """Sequential predictions made easier:
    - builds and stores GP models
    - stores the dataset (xi, zi) of size ni x d, ni x p
    - append new evaluations
    - make predicions
    - simulate conditional sample paths
    """

    def __init__(self,
                 dim_output=1,
                 models=None):
        """ properties initialization
        """

        # dataset
        # xi : ndarray(ni, d)
        # zi : ndarray(ni, dim_output)
        self.xi = None 
        self.zi = None

        # initialize models
        self.dim_output = dim_output
        self.build_models(models)

        # conditional_simulations
        self.n_samplepaths = None
        self.zsim = None
        self.xtsim = None

    def build_models(self, models):
        if models is None:
            self.models = [{'name': '',
                            'model': None,
                            'parameters_initial_guess': None,
                            'make_selection_criterion': None}] * self.dim_output

            for i in range(self.dim_output):
                self.models[i]['model'] = gp.core.Model(
                    self.constant_mean,
                    self.default_covariance,
                    None,
                    None)
                self.models[i]['parameters_initial_guess'] = gp.kernel.anisotropic_parameters_initial_guess
                self.models[i]['make_selection_criterion'] = gp.kernel.make_reml_criterion
        else:
            self.models = models
        
    def constant_mean(self, x, param):
        return np.ones((x.shape[0], 1))

    def default_covariance(self, x, y, covparam, pairwise=False):
        p = 2
        return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

    def set_data(self, xi, zi):
        self.xi = xi
        self.zi = zi.reshape(zi.shape[0], -1)  # self.zi is a matrix
        # even if zi is a vector

    def set_data_with_model_selection(self, xi, zi):
        self.set_data(xi, zi)
        self.update_params()

    def set_new_eval(self, xnew, znew):
        self.xi = np.vstack((self.xi, xnew))
        self.zi = np.vstack((self.zi.reshape(self.zi.shape[0], -1), znew))

    def set_new_eval_with_model_selection(self, xnew, znew):
        self.set_new_eval(xnew, znew)
        self.update_params()

    def update_params(self):
        """Parameter selection"""
        for i in range(self.dim_output):
            if self.models[i]['model'].covparam is None:
                covparam0 = self.models[i]['parameters_initial_guess'](
                    self.models[i]['model'],
                    self.xi,
                    self.zi[:, i])
            else:
                covparam0 = self.models[i]['model'].covparam

            nlrl, dnlrl = self.models[i]['make_selection_criterion'](
                self.models[i]['model'],
                self.xi,
                self.zi[:, i])

            self.models[i]['model'].covparam = gp.kernel.autoselect_parameters(
                covparam0,
                nlrl,
                dnlrl)

    def predict(self, xt):
        """Prediction"""
        zpm = np.empty((xt.shape[0], self.dim_output))
        zpv = np.empty((xt.shape[0], self.dim_output))
        for i in range(self.dim_output):
            zpm[:, i], zpv[:, i] = self.models[i]['model'].predict(
                self.xi,
                self.zi[:, i],
                xt)
        zpv = np.maximum(zpv, 0)
        return zpm, zpv

    def conditional_simulations(self, xt, n_samplepaths, enforce_unique=True):
        """Generate conditional sample paths"""

        # initialize sample paths
        self.n_samplepaths = n_samplepaths
        ni = self.xi.shape[0]
        nt = xt.shape[0]

        self.xtsim = np.vstack((self.xi, xt))

        if enforce_unique:
            self.xtsim, indices = np.unique(
                self.xtsim,
                return_inverse=True,
                axis=0)
            xi_ind = indices[0:ni]
            xt_ind = indices[ni:(ni+nt)]
            n = self.xtsim.shape[0]
        else:
            xi_ind = np.range(ni)
            xt_ind = np.range(nt) + ni
            n = ni + nt

        # unconditional sample paths
        self.zsim = np.empty(
            (n, n_samplepaths, self.dim_output))

        for i in range(self.dim_output):
            self.zsim[:, :, i] = self.models[i]['model'].sample_paths(
                self.xtsim,
                n_samplepaths)

        # conditional sample paths
        zpm = np.empty((nt, self.dim_output))
        zpv = np.empty((nt, self.dim_output))
        lambda_t = np.empty((ni, nt, self.dim_output))
        zpsim = np.empty((nt, n_samplepaths, self.dim_output))

        for i in range(self.dim_output):
            zpm[:, i], zpv[:, i], lambda_t[:, :, i] = self.models[i]['model'].predict(
                self.xi,
                self.zi[:, i],
                xt,
                return_lambdas=True)
            zpsim[:, :, i] = self.models[i]['model'].conditional_sample_paths(
                self.zsim[:, :, i],
                xi_ind,
                self.zi[:, i],
                xt_ind,
                lambda_t[:, :, i])

        if self.dim_output == 1:
            return zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))
        else:
            return zpsim
