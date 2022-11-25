'''Implement a sketch of the EI algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.sampcrit as sampcrit

## -- definition of a mono-objective problem

class Problem:

    def __init__(self):

        self.dim = 1
        self.p = 1
        self.box = [[-1], [1]]

    def eval(self, x):
        return gp.misc.testfunctions.twobumps(x)


## -- create initial dataset

problem = Problem()

nt = 2000
xt = gp.misc.designs.regulargrid(problem.dim, nt, problem.box)
zt = gp.misc.testfunctions.twobumps(xt)

ind = [100, 1000, 1600]
xi = xt[ind]

## -- initialize ei algorithm

eialgo = ei.ExpectedImprovement(problem)

eialgo.set_initial_design(xi)

## -- visualization


def plot(show=True, x=None, z=None):
    
    zpm, zpv = eialgo.predict(xt)
    ei = sampcrit.expected_improvement(-np.min(eialgo.zi), -zpm, zpv)
    pe = sampcrit.probability_excursion(-np.min(eialgo.zi), -zpm, zpv)

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
    fig.subplot(1)
    fig.plot(xt, zt, 'k', linewidth=0.5)
    if z is not None:
        fig.plot(x, z, 'b', linewidth=0.5)
    fig.plotdata(eialgo.xi, eialgo.zi)
    fig.plotgp(xt, zpm, zpv, colorscheme='simple')
    fig.ylabel('$z$')
    fig.title('Posterior GP')
    fig.subplot(2)
    fig.plot(xt, -ei, 'k', linewidth=0.5)
    fig.ylabel('EI')
    fig.subplot(3)
    fig.plot(xt, pe, 'k', linewidth=0.5)
    fig.plot(eialgo.smc.x, np.zeros(eialgo.smc.n), '.')
    fig.ylabel('Prob. excursion')
    fig.xlabel('x')
    if show:
        fig.show()

    return fig


plot()

# make n = 3 new evaluations
n = 3
for i in range(n):
    eialgo.step()
    plot(show=True)
