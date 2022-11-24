'''Implement a sketch of the EI algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.sequentialprediction as gpsp
import gpmpcontrib.sampcrit as gpsc
import gpmpcontrib.smc as gpsmc

## -- create initial dataset

dim = 1
nt = 2000
box = [[-1], [1]]
xt = gp.misc.designs.regulargrid(dim, nt, box)
zt = gp.misc.testfunctions.twobumps(xt)

ind = [100, 1000, 1600]
xi = xt[ind]
zi = zt[ind]

## -- compute predictive distributions

sp = gpsp.SequentialPrediction()
sp.set_data_with_model_selection(xi, zi)
zpm, zpv = sp.predict(xt)

## -- SMC step: compute a sample of points distributed according to the
#     probability of exercursion


def log_prob_excursion(x):
    log_prob_excur = np.full((x.shape[0], ), -np.inf)
    b = gpsc.isinbox(box, x)
    zpm, zpv = sp.predict(x[b])
    log_prob_excur[b] = np.log(
        np.maximum(1e-6,
                   gpsc.probability_excursion(-np.min(sp.zi),
                                                  -zpm,
                                                  zpv))).flatten()
    return log_prob_excur


smc = gpsmc.SMC(box)
smc.step(log_prob_excursion)

## -- visualization


def plot(show=True, x=None, z=None):
    ei = gpsc.expected_improvement(-np.min(sp.zi), -zpm, zpv)
    pe = gpsc.probability_excursion(-np.min(sp.zi), -zpm, zpv)

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
    fig.subplot(1)
    fig.plot(xt, zt, 'k', linewidth=0.5)
    if z is not None:
        fig.plot(x, z, 'b', linewidth=0.5)
    fig.plotdata(sp.xi, sp.zi)
    fig.plotgp(xt, zpm, zpv, colorscheme='simple')
    fig.ylabel('$z$')
    fig.title('Posterior GP')
    fig.subplot(2)
    fig.plot(xt, -ei, 'k', linewidth=0.5)
    fig.ylabel('EI')
    fig.subplot(3)
    fig.plot(xt, pe, 'k', linewidth=0.5)
    fig.plot(smc.x, np.zeros(smc.n), '.')
    fig.ylabel('Prob. excursion')
    fig.xlabel('x')
    if show:
        fig.show()

    return fig


plot()


def ei_step():

    # make a new evaluation
    zpm, zpv = sp.predict(smc.x)
    ei = gpsc.expected_improvement(-np.min(zi), -zpm, zpv)
    x_new = smc.x[np.argmax(ei)]
    z_new = gp.misc.testfunctions.twobumps(x_new)
    sp.set_new_eval_with_model_selection(x_new, z_new)

    # make an SMC step
    smc.step(log_prob_excursion)


# make n = 3 new evaluations
n = 3
for i in range(n):
    ei_step()
    zpm, zpv = sp.predict(xt)
    plot(show=True)

## -- plot conditional sample paths
n_samplepaths = 6
zpsim = sp.conditional_simulations(xt, n_samplepaths)
plot(show=True, x=xt, z=zpsim)


## -- plot the empirical distribution of the minimizer
n_samplepaths = 500
zpsim = sp.conditional_simulations(xt, n_samplepaths)

xstar_ind = np.argmin(zpsim, axis=0)
xstar = xt[xstar_ind]

plt.figure()
plt.hist(xstar, 100)
plt.show()
