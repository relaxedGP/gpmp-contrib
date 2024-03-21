# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
from gpmpcontrib import SubsetPointwiseCriterion


class Straddle(SubsetPointwiseCriterion):

    def __init__(self, t, *args, **kwargs):
        self.t = t
        super().__init__(*args, **kwargs)

    def build_criterion(self):
        def criterion(zpm, zpv):
            return 1.96 * gnp.sqrt(zpv) - gnp.abs(zpm - self.t)
        return criterion
