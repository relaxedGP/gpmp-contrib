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


class ExpectedImprovement(SequentialPrediction):
    def __init__(self, problem, model, options=None):
        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(model=model)

        # options
        self.options = self.set_options(options)

        # search space
        self.smc = self.init_smc(self.options["n_smc"])

        # ei values
        self.ei = None

        # minimum
        self.current_minimum = None

    def set_options(self, options):
        default_options = {"n_smc": 1000}
        default_options.update(options or {})
        
        return default_options

    def init_smc(self, n_smc):
        return SMC(self.computer_experiments_problem.input_box, n_smc)

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
        method = 3
        if method == 1:
            self.smc.step(
                logpdf_parameterized_function=self.log_prob_excursion,
                u_target=-self.current_minimum,
            )
        elif method == 2:
            self.smc.restart(
                logpdf_parameterized_function=self.log_prob_excursion,
                initial_threshold=-gnp.max(self.zi),
                final_threshold=-self.current_minimum,
                p0=0.8,
                debug=True
            )
        elif method == 3:
            self.smc.step_with_possible_restart(
                logpdf_parameterized_function=self.log_prob_excursion,
                initial_threshold=-gnp.max(self.zi),
                target_threshold=-self.current_minimum,
                min_ess_ratio=0.6,
                p0=0.6,
                debug=False
            )
            


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
        znew = self.computer_experiments_problem.eval(xnew)

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        self.current_minimum = gnp.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def step(self):
        # evaluate ei on the search space
        zpm, zpv = self.predict(self.smc.particles.x, convert_out=False)
        self.ei = sampcrit.expected_improvement(-self.current_minimum, -zpm, zpv)

        # make new evaluation
        x_new = self.smc.particles.x[gnp.argmax(gnp.asarray(self.ei))].reshape(1, -1)

        self.make_new_eval(x_new)
