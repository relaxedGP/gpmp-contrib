"""Sequential Prediction with Maximum MSE Sampling Strategy

This script demonstrates the use of the SequentialPrediction class for
the approximation of a test function using a maximum MSE sampling
strategy. It includes steps to set up a problem, create an initial
dataset, make predictions, and iteratively improve the model with new
data points chosen based on maximum MSE.

Imports:
- numpy for numerical operations
- gpmp and gpmpcontrib for Gaussian Process modeling and sequential
  prediction

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import numpy as np
import gpmp as gp
import gpmpcontrib as gpc

# Set interactive mode for plotting (set to True if interactive plotting is desired)
interactive = False


def visualize_results(xt, zt, xi, zi, zpm, zpv, xnew=None):
    """
    Visualize the results of the predictions and the dataset.

    Parameters:
    xt (ndarray): Test points
    zt (ndarray): True values at test points
    xi (ndarray): Input data points
    zi (ndarray): Output values at input data points
    zpm (ndarray): Posterior mean values
    zpv (ndarray): Posterior variances
    xnew (ndarray, optional): New data point being added
    """
    fig = gp.misc.plotutils.Figure(isinteractive=interactive)
    fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    if xnew is not None:
        fig.plot(np.repeat(xnew, 2), fig.ylim(), color="tab:gray", linewidth=2)
    fig.xylabels("$x$", "$z$")
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


# -- Definition of a problem --

# Create a ComputerExperiment problem instance using the twobumps test function
problem = gpc.ComputerExperiment(
    1,  # Dimension of input domain
    [[-1], [1]],  # Input box (domain)
    single_function=gp.misc.testfunctions.twobumps,
)

# -- Create initial dataset --

# Generate a regular grid of test points within the input domain
nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

# Select a few initial data points and their corresponding outputs
ni = 3
ind = [100, 1000, 1600]
xi = xt[ind]
zi = problem(xi)

# -- Create SequentialPrediction object --

# Initialize SequentialPrediction with the initial dataset
model = gpc.Model(
    "Simple function",
    output_dim=problem.output_dim,
    mean="constant",
    covariance_params={"p": 2},
)
spred = gpc.SequentialPrediction(model)
spred.set_data_with_model_selection(xi, zi)

# Predict at the test points and visualize the results
zpm, zpv = spred.predict(xt)
visualize_results(xt, zt, xi, zi, zpm, zpv)


# Function for selecting new data points based on maximum MSE
def mmse_sampling(seqpred, xt):
    """
    Select a new data point for evaluation based on maximum MSE.

    Parameters:
    seqpred (SequentialPrediction): The sequential prediction model
    xt (ndarray): Test points

    Returns:
    ndarray: The new data point selected for evaluation
    """
    zpm, zpv = seqpred.predict(xt)
    maxmse_ind = np.argmax(zpv)
    xi_new = xt[maxmse_ind]
    return xi_new.reshape(-1, 1)


# -- Iterative model improvement --

# Number of iterations for model improvement
n = 10
for i in range(n):
    # Select a new data point and visualize
    xi_new = mmse_sampling(spred, xt)
    visualize_results(xt, zt, spred.xi, spred.zi, zpm, zpv, xi_new)

    # Evaluate the new data point and update the model
    zi_new = problem(xi_new)
    spred.set_new_eval_with_model_selection(xi_new, zi_new)
    zpm, zpv = spred.predict(xt)

# Visualize the final results
visualize_results(xt, zt, spred.xi, spred.zi, zpm, zpv)
