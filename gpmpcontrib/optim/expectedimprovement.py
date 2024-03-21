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


class ExpectedImprovement(SequentialPrediction):
    def __init__(self, problem, model, options=None):
        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(model=model)

        # options
        self.options = self.set_options(options)

        # search space
        self.smc = self.init_smc()

        # ei values
        self.ei = None

        # minimum
        self.current_minimum = None

    def set_options(self, options):
        default_options = {
            "smc_options": {"n": 1000, "mh_params": {}},
            "smc_method": "step_with_possible_restart"
        }
        ExpectedImprovement.deep_update(default_options, options or {})

        return default_options
    @staticmethod
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                if k in d.keys():
                    d_k = d.get(k)
                    if isinstance(d_k, collections.Mapping):
                        ExpectedImprovement.deep_update(d_k, v)
                        continue
            d[k] = v

    def init_smc(self):
        return SMC(self.computer_experiments_problem.input_box, **self.options["smc_options"])

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
        if method == "step_simple":
            self.smc.step(
                logpdf_parameterized_function=self.log_prob_excursion,
                logpdf_param=-self.current_minimum,
            )
        elif method == "restart":
            self.smc.restart(
                logpdf_parameterized_function=self.log_prob_excursion,
                logpdf_initial_param=-gnp.max(self.zi),
                target_logpdf_param=-self.current_minimum,
                p0=0.8,
                debug=False
            )
        elif method == "step_with_possible_restart":
            self.smc.step_with_possible_restart(
                logpdf_parameterized_function=self.log_prob_excursion,
                logpdf_initial_param=-gnp.max(self.zi),
                target_logpdf_param=-self.current_minimum,
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

        self.current_minimum = gnp.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        znew = self.computer_experiments_problem.eval(xnew.numpy())

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        self.current_minimum = gnp.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def local_ei_opt(self, init):
        """
            init : ndarray
        Initial guess of the EI maximizer.
        """

        def crit_(x):
            x_row = x.reshape(1, -1)

            zpm, zpv = self.predict(x_row, convert_out=False)
            ei = sampcrit.expected_improvement(-self.current_minimum, -zpm, zpv)

            return - ei[0, 0]

        crit_jit = gnp.jax.jit(crit_)

        dcrit = gnp.jax.jit(gnp.grad(crit_jit))

        box = self.computer_experiments_problem.input_box
        assert all([len(_v) == len(box[0]) for _v in box])

        bounds = [tuple(box[i][k] for i in range(len(box))) for k in range(len(box[0]))]
        ei_argmax = gp.kernel.autoselect_parameters(
            init, crit_jit, dcrit, bounds=bounds
        )

        if gnp.numpy.isnan(ei_argmax).any():
            return init

        for i in range(ei_argmax.shape[0]):
            if ei_argmax[i] < bounds[i][0]:
                ei_argmax[i] = bounds[i][0]
            if bounds[i][1] < ei_argmax[i]:
                ei_argmax[i] = bounds[i][1]

        if crit_(ei_argmax) < crit_(init):
            output = ei_argmax
        else:
            output = init

        return gnp.asarray(output.reshape(1, -1))

    def step(self):
        # evaluate ei on the search space
        zpm, zpv = self.predict(self.smc.particles.x, convert_out=False)
        self.ei = sampcrit.expected_improvement(-self.current_minimum, -zpm, zpv)

        assert not gnp.isnan(self.ei).any()

        # make new evaluation
        x_new = self.smc.particles.x[gnp.argmax(gnp.asarray(self.ei))].reshape(1, -1)

        x_new = self.local_ei_opt(gnp.to_np(x_new).ravel())

        self.make_new_eval(x_new)
