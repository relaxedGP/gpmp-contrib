import numpy as np
from gpmpcontrib.computerexperiment import ComputerExperiment
import gpmpcontrib.optim.test_problems as optim_problems

def __getattr__(name):
    if name == "g10c6":
        return g10c6
    elif name == "g10c6mod":
        return g10c6mod
    elif name == "g10c6modmod":
        return g10c6modmod

    name_split = name.split("-")
    assert len(name_split) == 2, name
    name_base = name_split[0]
    try:
        t = float(name_split[1])
    except ValueError:
        raise ValueError(name)

    if name_base == "goldsteinprice":
        parametrized_problem = goldsteinprice
    elif name_base == "goldstein_price_log":
        parametrized_problem = goldstein_price_log
    elif name_base == "branin":
        parametrized_problem = branin
    else:
        raise ValueError(name)

    return parametrized_problem(t)


# ===== G10C6 =====
_g10c6_constraints = lambda x: optim_problems._g10_constraints(x)[:, 5]

_g10c6_dict = optim_problems._g10_dict.copy()
_g10c6_dict["single_objective"] = None
_g10c6_dict["single_constraint"] = {"function": _g10c6_constraints, "output_dim": 1, "bounds": [[-np.inf, 0]]}

g10c6 = ComputerExperiment(
    _g10c6_dict["input_dim"],
    _g10c6_dict["input_box"],
    single_objective=_g10c6_dict["single_objective"],
    single_constraint=_g10c6_dict["single_constraint"]
)

# ===== G10C6MOD =====
_g10c6mod_constraints = lambda x: optim_problems._g10mod_constraints(x)[:, 5]

_g10c6mod_dict = optim_problems._g10mod_dict.copy()
_g10c6mod_dict["single_objective"] = None
_g10c6mod_dict["single_constraint"] = {"function": _g10c6mod_constraints, "output_dim": 1, "bounds": [[-np.inf, 0]]}

g10c6mod = ComputerExperiment(
    _g10c6mod_dict["input_dim"],
    _g10c6mod_dict["input_box"],
    single_objective=_g10c6mod_dict["single_objective"],
    single_constraint=_g10c6mod_dict["single_constraint"]
)

# ===== G10C6MODMOD =====
_g10c6modmod_constraints = lambda x: optim_problems._g10modmod_constraints(x)[:, 5]

_g10c6modmod_dict = optim_problems._g10modmod_dict.copy()
_g10c6modmod_dict["single_objective"] = None
_g10c6modmod_dict["single_constraint"] = {"function": _g10c6modmod_constraints, "output_dim": 1, "bounds": [[-np.inf, 0]]}

g10c6modmod = ComputerExperiment(
    _g10c6modmod_dict["input_dim"],
    _g10c6modmod_dict["input_box"],
    single_objective=_g10c6modmod_dict["single_objective"],
    single_constraint=_g10c6modmod_dict["single_constraint"]
)


# ==== GoldsteinPrice function ====
_goldsteinprice_dict = optim_problems._goldsteinprice_dict.copy()
_goldsteinprice_dict["single_objective"] = None

goldsteinprice = lambda t: ComputerExperiment(
    _goldsteinprice_dict["input_dim"],
    _goldsteinprice_dict["input_box"],
    single_constraint={
        "function": optim_problems._goldsteinprice_objective,
        "output_dim": 1,
        "bounds": [[-np.inf, t]]
    }
)


#  ==== log-GoldsteinPrice function ====
_goldstein_price_log_dict = optim_problems._goldstein_price_log_dict.copy()
_goldstein_price_log_dict["single_objective"] = None

goldstein_price_log = lambda t: ComputerExperiment(
    _goldstein_price_log_dict["input_dim"],
    _goldstein_price_log_dict["input_box"],
    single_constraint={
        "function": optim_problems._goldstein_price_log_objective,
        "output_dim": 1,
        "bounds": [[-np.inf, t]]
    }
)


# ===== Branin
_branin_dict = optim_problems._branin_dict.copy()
_branin_dict["single_objective"] = None

branin = lambda t: ComputerExperiment(
    _branin_dict["input_dim"],
    _branin_dict["input_box"],
    single_constraint={
        "function": optim_problems._branin_objective,
        "output_dim": 1,
        "bounds": [[-np.inf, t]]
    }
)
