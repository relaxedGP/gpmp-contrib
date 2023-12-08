# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import scipy.stats as stats
import gpmp.num as gnp
import gpmp as gp


class SMC:
    '''SMC object: builds a set of particles targeting a
       distribution given by a log-pdf

    '''

    def __init__(self, box, n=1000):

        self.n = n
        (self.x, self.logpx, self.w) = self.particles_init(box, n)

        self.logpdf = None

        self.param_s = 0.05

    def particles_init(self, box, n):
        dim = len(box[0])
        x = gnp.asarray(gp.misc.designs.maximinlhs(dim, n, box))
        logpx = gnp.zeros((n,))
        w = gnp.full((n, ), 1 / n)

        return (x, logpx, w)

    def set_logpdf(self, logpdf):
        self.logpdf = logpdf

    def reweight(self):
        logpx_new = self.logpdf(self.x)
        self.w = self.w * gnp.exp(logpx_new - self.logpx)
        self.logpx = logpx_new

    def ess(self):
        '''https://en.wikipedia.org/wiki/Effective_sample_size'''
        return gnp.sum(self.w)**2 / gnp.sum(self.w**2)

    def resample(self):
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        try:
            counts = stats.multinomial.rvs(self.n, p)
        except:
            extype, value, tb = __import__("sys").exc_info()
            __import__("traceback").print_exc()
            __import__("pdb").post_mortem(tb)

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled = gnp.set_row2(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem1(logpx_resampled, i, self.logpx[j])
                counts[j] -= 1
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n, ), 1 / self.n)

    def pertubate(self):
        C = self.param_s * gnp.cov(self.x.reshape(self.x.shape[0], -1).T)
        eps = gnp.multivariate_normal.rvs(cov=C, n=self.n)

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        y = self.pertubate()
        logpy = self.logpdf(y)

        rho = gnp.minimum(1, gnp.exp(logpy - self.logpx))

        for i in range(self.n):
            if gnp.rand(1) < rho[i]:
                self.x = gnp.set_row2(self.x, i, y[i, :])
                self.logpx = gnp.set_elem1(self.logpx, i, logpy[i])
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(self.x, self.logpx, '.'); plt.show()

    def step(self, logpdf):
        self.set_logpdf(logpdf)
        self.reweight()
        self.resample()
        self.move()
