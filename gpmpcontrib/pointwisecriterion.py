# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import SequentialPrediction
from gpmpcontrib import SMC
import collections


class PointwiseCriterion(SequentialPrediction):
    def __init__(self, problem, model, options=None):
        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(model=model)

        # options
        self.options = self.set_options(options)

        # search space
        self.smc = self.init_smc()

        # criterion values
        self.criterion_values = None

        # criterion
        self.criterion = self.build_criterion()

    def set_options(self, options):
        default_options = {
            "smc_options": {"n": 1000, "mh_params": {}},
            "smc_method": "step_with_possible_restart"
        }
        PointwiseCriterion.deep_update(default_options, options or {})

        return default_options

    @staticmethod
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                if k in d.keys():
                    d_k = d.get(k)
                    if isinstance(d_k, collections.abc.Mapping):
                        PointwiseCriterion.deep_update(d_k, v)
                        continue
            d[k] = v

    def init_smc(self):
        return SMC(self.computer_experiments_problem.input_box, **self.options["smc_options"])

    # Useful for several subclasses
    def log_prob_excursion(self, x, u):
        min_threshold = 1e-6  # we do not want -inf values in the log probability
        sigma2_scale_factor = (
            2.0**2  # increase exploration by increasing predictive variance
        )

        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        b = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False)

        log_prob_excur = gnp.where(
            gnp.asarray(b),
            gnp.log(
                gnp.maximum(
                    min_threshold,
                    sampcrit.probability_excursion(u, -zpm, sigma2_scale_factor * zpv),
                )
            ).flatten(),
            -gnp.inf,
        )

        return log_prob_excur

    def update_search_space(self):
        method = self.options["smc_method"]

        logpdf_parameterized_function, logpdf_initial_param, target_logpdf_param = self.logpdf_parameterized()

        if method == "step_simple":
            self.smc.step(
                logpdf_parameterized_function=logpdf_parameterized_function,
                logpdf_param=target_logpdf_param,
            )
        elif method == "restart":
            self.smc.restart(
                logpdf_parameterized_function=logpdf_parameterized_function,
                logpdf_initial_param=logpdf_initial_param,
                target_logpdf_param=target_logpdf_param,
                p0=0.8,
                debug=False
            )
        elif method == "step_with_possible_restart":
            self.smc.step_with_possible_restart(
                logpdf_parameterized_function=logpdf_parameterized_function,
                logpdf_initial_param=logpdf_initial_param,
                target_logpdf_param=target_logpdf_param,
                min_ess_ratio=0.6,
                p0=0.6,
                debug=False
            )
        else:
            raise ValueError(method)

    def set_initial_design(self, xi, update_model=True, update_search_space=True):
        zi = self.computer_experiments_problem.eval(xi)

        if update_model:
            super().set_data_with_model_selection(xi, zi)
        else:
            super().set_data(xi, zi)

        if update_search_space:
            self.update_search_space()

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        znew = self.computer_experiments_problem.eval(xnew.numpy())

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        if update_search_space:
            self.update_search_space()

    def local_criterion_opt(self, init):
        """
            init : ndarray
        Initial guess of the criterion maximizer.
        """

        def crit_(x):
            x_row = x.reshape(1, -1)

            zpm, zpv = self.predict(x_row, convert_out=False)
            criterion_value = self.criterion(zpm, zpv)

            return - criterion_value[0, 0]

        crit_jit = gnp.jax.jit(crit_)

        dcrit = gnp.jax.jit(gnp.grad(crit_jit))

        box = self.computer_experiments_problem.input_box
        assert all([len(_v) == len(box[0]) for _v in box])

        bounds = [tuple(box[i][k] for i in range(len(box))) for k in range(len(box[0]))]
        criterion_argmax = gp.kernel.autoselect_parameters(
            init, crit_jit, dcrit, bounds=bounds
        )

        if gnp.numpy.isnan(criterion_argmax).any():
            return init

        for i in range(criterion_argmax.shape[0]):
            if criterion_argmax[i] < bounds[i][0]:
                criterion_argmax[i] = bounds[i][0]
            if bounds[i][1] < criterion_argmax[i]:
                criterion_argmax[i] = bounds[i][1]

        if crit_(criterion_argmax) < crit_(init):
            output = criterion_argmax
        else:
            output = init

        return gnp.asarray(output.reshape(1, -1))

    def step(self):
        # evaluate the criterion on the search space
        zpm, zpv = self.predict(self.smc.particles.x, convert_out=False)
        self.criterion_values = self.criterion(zpm, zpv)

        assert not gnp.isnan(self.criterion_values).any()

        # make new evaluation
        x_new = self.smc.particles.x[gnp.argmax(gnp.asarray(self.criterion_values))].reshape(1, -1)

        x_new = self.local_criterion_opt(gnp.to_np(x_new).ravel())

        self.make_new_eval(x_new)
