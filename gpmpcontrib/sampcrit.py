'''Helper functions for sequential designs

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''

import numpy as np
import scipy.stats as stats


def isinbox(box, x):
    b = np.logical_and(np.all(x >= box[0], axis=1),
                       np.all(x <= box[1], axis=1))
    return b


def probability_excursion(t, zpm, zpv):
    '''Computes the probabilities of exceeding the threshold t for
    Gaussian predictive distributions with means zpm and variances
    zpv. The input argument must have the following sizes:

        * zpm        M x 1,
        * zpv        M x 1,
 
     where M is the number of points where the EI must be
     computed. The output has size M x 1.
    '''
    p = np.empty(zpm.shape)
    delta = zpm - t
    sigma = np.sqrt(zpv)
    b = sigma > 0

    # Compute p where sigma > 0
    u = delta[b] / sigma[b]
    p[b] = stats.norm.cdf(u)

    # Compute p where sigma == 0
    b = np.logical_not(b)
    p[b] = delta[b] > 0

    return p

def probability_box(box, zpm, zpv):
    '''Computes the probability to be in a box
    '''
    dim_output = zpm.shape[1]

    pn = np.empty(zpm.shape)
    for j in range(dim_output):

        delta_min = box[0][j] - zpm[:, j] 
        delta_max = box[1][j] - zpm[:, j]
        sigma = np.sqrt(zpv)
        b = sigma > 0

        # Compute pn where sigma > 0
        u_min = delta_min[b] / sigma[b]
        u_max = delta_max[b] / sigma[b]
        pn[b] = stats.norm.cdf(u_max) - stats.norm.cdf(u_min)

        # Compute p where sigma == 0
        b = np.logical_not(b)
        pn[b] = delta_max[b] > 0 & delta_min[b] < 0

        return pn

def expected_improvement(t, zpm, zpv):
    '''Computes the Expected Improvement (EI) criterion for a
     maximization problem given a threshold t and Gaussian predictive
     distributions with means zpm and variances zpv. The input
     argument must have the following sizes:
 
        * zpm        M x 1,
        * zpv        M x 1,
 
     where M is the number of points where the EI must be
     computed. The output has size M x 1.
 
     REFERENCES
 
    [1] D. R. Jones, M. Schonlau and William J. Welch. Efficient global
        optimization of expensive black-box functions.  Journal of Global
        Optimization, 13(4):455-492, 1998.
 
    [2] J. Mockus, V. Tiesis and A. Zilinskas. The application of Bayesian
        methods for seeking the extremum. In L.C.W. Dixon and G.P. Szego,
        editors, Towards Global Optimization, volume 2, pages 117-129, North
        Holland, New York, 1978.

    '''
    ei = np.empty(zpm.shape)
    delta = zpm - t
    sigma = np.sqrt(zpv)
    b = sigma > 0

    # Compute the EI where sigma > 0
    u = delta[b] / sigma[b]
    ei[b] = sigma[b] * (stats.norm.pdf(u) + u * stats.norm.cdf(u))

    # Compute the EI where sigma == 0
    b = np.logical_not(b)
    ei[b] = np.maximum(0, delta[b])

    # Correct numerical inaccuracies
    ei[ei < 0] = 0

    return ei
