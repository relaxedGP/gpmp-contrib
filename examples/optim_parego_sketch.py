# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.sequentialprediction as spred
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.optim.pareto as pareto

## -- definition of a bi-objective problem
class Problem:

    def __init__(self):

        self.input_dim = 2
        self.output_dim = 2
        self.input_box = [[0, 0], [1, 1]]

        self.m1 = [0.3, 0.8]
        self.c1 = [
            7.8e-01, 6.0, -4.7, 9.0e+01, -8.5e+01, -8.2e+01, 6.0e+02, 8.9e+02,
            3.7e+02, -7.4e+02
        ]
        self.m2 = [0.6, 0.6]
        self.c2 = [
            -4.5e-01, 7.8, -7.7, 2.8e+01, 3.4e+01, -3.1e+01, -5.0e+02,
            -1.7e+02, -4.8e+02, 5.3e+02
        ]

    def f(self, c, m, x):
        x = x - m
        z = c[0] + c[1] * x[:, 0] + c[2] * x[:, 1] + c[3] * x[:, 0] * x[:, 1] \
        + c[4] * x[:, 0] * x[:, 0] + c[5] * x[:, 1] * x[:, 1] \
        + c[6] * x[:, 0] * x[:, 0] * x[:, 1] + c[7] * x[:, 1] * x[:, 1] * x[:, 0] \
        + c[8] * x[:, 0] * x[:, 0] * x[:, 0] + c[9] * x[:, 1] * x[:, 1] * x[:, 1]

        return z

    def eval(self, x):
        n = x.shape[0]
        z1 = self.f(self.c1, self.m1, x).reshape((n, 1))
        z2 = self.f(self.c2, self.m2, x).reshape((n, 1))
        z = np.hstack((z1, z2))
        return z


## -- Plot Pareto front
objectives_num = 2
problem = Problem()

nt = [40, 40]  # Size of the regular grid
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
# nt = 500
# xt = gp.misc.designs.ldrandunif(problem.input_dim, nt, problem.input_box)

zt = problem.eval(xt)
zt_opt_b = pareto.pareto_points(zt)

def figure01():
    plt.plot(zt[:, 0], zt[:, 1], 'bo', markersize=2)
    pareto.plot_pareto(plt.gca(), zt[zt_opt_b])
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.title('Pareto front')
    plt.show()


figure01()

## -- initial design & predictions
ni = 8
xi = gp.misc.designs.maximinldlhs(problem.input_dim, ni, problem.input_box)
zi = problem.eval(xi)

sp = spred.SequentialPrediction(output_dim=objectives_num)
sp.set_data_with_model_selection(xi, zi)
zpm, zpv = sp.predict(xt)

def figure02():
    # -- predictions vs truth
    fig, axes = plt.subplots(nrows=1, ncols=2)

    for i in range(objectives_num):
        axes[i].plot(zt[:, i], zpm[:, i], 'ko')
        (xmin, xmax), (ymin, ymax) = axes[i].get_xlim(), axes[i].get_ylim()
        xmin = min(xmin, ymin)
        xmax = max(xmax, ymax)
        axes[i].plot([xmin, xmax], [xmin, xmax], '--')
        axes[i].set_xlabel('true values')
        axes[i].set_ylabel('predictions')
        axes[i].set_title("$f_{}$".format(i + 1))
    plt.show()


figure02()

def figure03():
    # -- conditional Pareto fronts
    nsim = 6
    sp.compute_conditional_simulations(n_samplepaths=nsim, xt=xt)
    plt.plot(zt[:, 0], zt[:, 1], 'bo', markersize=2)
    plt.plot(zi[:, 0], zi[:, 1], 'o', markersize=5, color='lime')
    pareto.plot_pareto(plt.gca(), zt[zt_opt_b])
    
    for i in range(nsim):
        zpsim_opt_b = pareto.pareto_points(sp.zpsim[:, i, :])
        pareto.plot_pareto(plt.gca(), sp.zpsim[zpsim_opt_b, i, :], color='orange')
    zpm, zpv = sp.predict(xt)
    zpm_opt_b = pareto.pareto_points(zpm)

    pareto.plot_pareto(plt.gca(), zpm[zpm_opt_b], color='magenta')
    plt.title('Posterior Pareto fronts')
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.show()


figure03()

## -- Optimization by scalarization

output_box = np.vstack( (np.min(zt, axis=0), np.max(zt, axis=0)) )

def rescale(z, box):
    """rescale outputs"""
    z_ = np.empty(z.shape)
    z_[:, 0] = (z[:, 0] - box[0, 0])/(box[1, 0] - box[0, 0])
    z_[:, 1] = (z[:, 1] - box[0, 1])/(box[1, 1] - box[0, 1])
    return z_
    
def f_w(w, z):
    """augmented Tchebychev scalarization function"""
    rho = 0.05
    return np.max(w*z, axis=1) + rho * np.sum(np.abs(w * z), axis=1)

# weight vector
w0 = np.array([0, 1])

# aggregated objective
zi_w0 = f_w(w0, rescale(zi, output_box))

# build model for aggregated objective
sp_w = spred.SequentialPrediction(output_dim=1)
sp_w.set_data_with_model_selection(xi, zi_w0)

# ei step on aggregated objective
cmap = plt.get_cmap('PiYG')
contour_lines = 30

def ei_step():
    # compute ei
    zpm_w, zpv_w = sp_w.predict(xt)
    ei = sampcrit.expected_improvement(-np.min(sp_w.zi), -zpm_w, zpv_w)
    plt.contourf(xt[:, 0].reshape(nt), xt[:, 1].reshape(nt), ei.reshape(nt), levels=contour_lines, cmap=cmap)
    plt.title('EI values in search space, ni = {}'.format(sp.xi.shape[0]))
    plt.colorbar()
    plt.show()
    # make a new eval
    x_new = xt[np.argmax(ei)].reshape((1, 2))
    z_new = problem.eval(x_new)
    z_new_w = f_w(w0, rescale(z_new, output_box))
    sp.set_new_eval_with_model_selection(x_new, z_new)
    sp_w.set_new_eval_with_model_selection(x_new, z_new_w)

n = 3
for i in range(n):
    ei_step()

def figure04():
    zpm, zpv = sp.predict(xt)
    # -- conditional Pareto fronts
    nsim = 6
    sp.compute_conditional_simulations(n_samplepaths=nsim, xt=xt)
    plt.plot(zt[:, 0], zt[:, 1], 'bo', markersize=2)
    pareto.plot_pareto(plt.gca(), zt[zt_opt_b])
    for i in range(nsim):
        zpsim_opt_b = pareto.pareto_points(sp.zpsim[:, i, :])
        pareto.plot_pareto(plt.gca(), sp.zpsim[zpsim_opt_b, i, :], color='orange')
    zpm_opt_b = pareto.pareto_points(zpm)
    pareto.plot_pareto(plt.gca(), zpm[zpm_opt_b], color='magenta')
    plt.plot(sp.zi[:, 0], sp.zi[:, 1], 'o', markersize=5, color='lime')
    plt.title('Posterior Pareto fronts')
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.show()

figure04()

## -- optimization in a new direction
w0 = np.array([1, 0])

zi_w0 = f_w(w0, rescale(zi, output_box))

sp_w = spred.SequentialPrediction(output_dim=1)
sp_w.set_data_with_model_selection(xi, zi_w0)

n = 3
for i in range(n):
    ei_step()
figure04()
