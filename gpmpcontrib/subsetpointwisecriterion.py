# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import PointwiseCriterion


class SubsetPointwiseCriterion(PointwiseCriterion):

    def get_target(self):
        zpm_xi, zpv_xi = self.predict(self.xi, convert_out=False)
        criterion_xi = self.criterion(zpm_xi, zpv_xi)

        zpm_particles, zpv_particles = self.predict(self.smc.particles.x, convert_out=False)
        criterion_particles = self.criterion(zpm_particles, zpv_particles)

        target = max(criterion_xi.max(), criterion_particles.max())

        return target

    def boxify_criterion(self, x):
        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        b = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False)

        res = self.criterion(zpm, zpv).flatten()

        res = gnp.where(gnp.asarray(b), res, - gnp.inf)

        return res

    def update_search_space(self):
        method = self.options["smc_method"]

        target = self.get_target()

        if method == "subset":
            self.smc.subset(
                func=self.boxify_criterion,
                target=target,
                p0=0.2,
                xi=self.xi,
                debug=False
            )
        else:
            raise ValueError(method)
