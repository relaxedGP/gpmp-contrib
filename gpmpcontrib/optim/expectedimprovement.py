# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import numpy as np
import gpmp as gp
import gpmpcontrib.sequentialprediction as spred
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.smc as gpsmc


class ExpectedImprovement(spred.SequentialPrediction):

    def __init__(self, problem, model=None, options=None):

        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(output_dim=1, models=model)

        # options
        self.options = self.set_options(options)

        # search space
        self.smc = self.init_smc(self.options['n_smc'])

        # ei values
        self.ei = None

        # minimum
        self.minimum = None

    def set_options(self, options):
        default_options = {'n_smc': 1000}
        return default_options

    def init_smc(self, n_smc):
        return gpsmc.SMC(self.computer_experiments_problem.input_box, n_smc)

    def log_prob_excursion(self, x):
        min_threshold = 1e-6
        log_prob_excur = np.full((x.shape[0], ), -np.inf)
        b = sampcrit.isinbox(self.computer_experiments_problem.input_box, x)

        zpm, zpv = self.predict(x[b])

        log_prob_excur[b] = np.log(
            np.maximum(
                min_threshold,
                sampcrit.probability_excursion(
                    -np.min(self.zi),
                    -zpm,
                    zpv
                )
            )
        ).flatten()

        return log_prob_excur

    def update_search_space(self):
        self.smc.step(self.log_prob_excursion)

    def set_initial_design(self, xi, update_model=True, update_search_space=True):
        zi = self.computer_experiments_problem.eval(xi)

        if update_model:
            super().set_data_with_model_selection(xi, zi)
        else:
            super().set_data(xi, zi)

        self.minimum = np.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        znew = self.computer_experiments_problem.eval(xnew)

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        self.minimum = np.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def step(self):
        # evaluate ei on the search space
        zpm, zpv = self.predict(self.smc.x)
        self.ei = sampcrit.expected_improvement(-self.minimum, -zpm, zpv)

        # make new evaluation
        x_new = self.smc.x[np.argmax(self.ei)]
        self.make_new_eval(x_new)
