import numpy as np
from gpmpcontrib.computerexperiment import ComputerExperiment
import gpmpcontrib.optim.test_problems as optim_problems

def __getattr__(name):
    if name == "c6":
        return c6
    elif name == "c67":
        return c67

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

# ===== C6 =====
_c6_constraints = lambda x: optim_problems._g10RR_constraints(x)[:, 5]

_c6_dict = optim_problems._g10RR_dict.copy()
_c6_dict["single_objective"] = None
_c6_dict["single_constraint"] = {"function": _c6_constraints, "output_dim": 1, "bounds": [[-np.inf, 0]]}

c6 = ComputerExperiment(
    _c6_dict["input_dim"],
    _c6_dict["input_box"],
    single_objective=_c6_dict["single_objective"],
    single_constraint=_c6_dict["single_constraint"]
)

# ===== C67 =====
_c67_constraints = lambda x: optim_problems._g10RRmod_constraints(x)[:, 5]

_c67_dict = optim_problems._g10RRmod_dict.copy()
_c67_dict["single_objective"] = None
_c67_dict["single_constraint"] = {"function": _c67_constraints, "output_dim": 1, "bounds": [[-np.inf, 0]]}

c67 = ComputerExperiment(
    _c67_dict["input_dim"],
    _c67_dict["input_box"],
    single_objective=_c67_dict["single_objective"],
    single_constraint=_c67_dict["single_constraint"]
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
