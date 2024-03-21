# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
from gpmpcontrib import SubsetPointwiseCriterion
import scipy.stats


class UCB(SubsetPointwiseCriterion):

    def __init__(self, q, *args, **kwargs):
        self.q = q
        self.alpha = scipy.stats.norm.ppf(self.q)
        super().__init__(*args, **kwargs)

    def build_criterion(self):
        def criterion(zpm, zpv):
            return - zpm - self.alpha * gnp.sqrt(zpv)
        return criterion

