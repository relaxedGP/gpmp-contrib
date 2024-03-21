# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import PointwiseCriterion


class ExpectedImprovement(PointwiseCriterion):

    def build_criterion(self):
        def criterion(zpm, zpv):
            return sampcrit.expected_improvement(-gnp.min(self.zi), -zpm, zpv)
        return criterion

    def logpdf_parameterized(self):
        logpdf_parameterized_function = self.log_prob_excursion
        logpdf_initial_param = -gnp.max(self.zi)
        target_logpdf_param = -gnp.min(self.zi)
        return logpdf_parameterized_function, logpdf_initial_param, target_logpdf_param
