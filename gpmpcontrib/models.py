"""GP models

This script contains the implementation of Gaussian Process (GP)
models. It includes functions for building custom kernels, setting
mean and covariance functions, and initial guess procedures for model
parameters. It also includes support for multi-output modeling and
noise handling

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import time
import gpmp.num as gnp
import gpmp as gp
from math import log
import gpmpcontrib.regp as regp

# ==============================================================================
# Model Class
# ==============================================================================


class Model:
    def __init__(
        self,
        name,
        output_dim,
        parameterized_mean,
        mean_params,
        covariance_params,
        box,
        rng,
        initial_guess_procedures=None,
        selection_criteria=None,
    ):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        parameterized_mean : bool
            If True, the model uses a "parameterized" mean function. If False,
            it uses a "linear predictor" mean function.
        mean_params : list of dict or dict
            Parameters for defining mean functions. Can be a single dictionary applied to
            all outputs or a list of dictionaries, one for each output. Each dictionary includes:
            - 'function': The mean function to be used, either a callable or a string.
            - 'param_length': The length of the mean function's parameter vector, required if 'function' is a callable.
        covariance_params : dict or list of dicts
            Parameters for defining covariance functions. Each dictionary must include a key 'function'.
        box : array_like
            The domain box.
        rng : numpy.random.Generator
            Random number generator.
        initial_guess_procedures : list of callables, optional
            A list of procedures for initial guess of model parameters, one for each output.
        selection_criteria : list of callables, optional
            A list of selection criteria, one for each output.
        """
        self.name = name
        self.output_dim = output_dim
        self.box = box
        self.rng = rng

        self.parameterized_mean = parameterized_mean
        if self.parameterized_mean:
            mean_type = "parameterized"
        else:
            mean_type = "linear_predictor"

        self.mean_functions, self.mean_functions_info = self.set_mean_functions(
            mean_params
        )
        self.covariance_functions = self.set_covariance_functions(covariance_params)

        # Initialize the models
        self.models = []
        for i in range(output_dim):
            model = gp.core.Model(
                self.mean_functions[i],
                self.covariance_functions[i],
                meanparam=None,
                covparam=None,
                meantype=mean_type,
            )
            self.models.append(
                {
                    "output_name": f"output{i}",
                    "model": model,
                    "mean_fname": self.mean_functions_info[i]["description"],
                    "mean_paramlength": self.mean_functions_info[i]["param_length"],
                    "covariance_fname": model.covariance.__name__,
                    "parameters_initial_guess_procedure": None,
                    "pre_selection_criterion": None,
                    "info": None,
                }
            )

        # Set initial guess procedures and selection criteria after model initialization
        parameters_initial_guess_procedures = (
            self.set_parameters_initial_guess_procedures(initial_guess_procedures)
        )
        selection_criteria = self.set_selection_criteria(selection_criteria)

        # Assign initial guess procedures and selection criteria to models
        for i in range(self.output_dim):
            self.models[i][
                "parameters_initial_guess_procedure"
            ] = parameters_initial_guess_procedures[i]
            self.models[i]["pre_selection_criterion"] = selection_criteria[i]

    def __getitem__(self, index):
        """
        Allows accessing the individual models and their attributes using the index.

        Parameters
        ----------
        index : int
            The index of the model to access.

        Returns
        -------
        dict
            The dictionary containing the model and its associated attributes.
        """
        return self.models[index]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the Model instance.
        """
        model_info = f"Model Name: {self.name}, Output Dimension: {self.output_dim}\n"
        for i, model in enumerate(self.models):
            mean_descr = self.mean_functions_info[i]["description"]
            mean_type = model["model"].meantype
            mean_params = model["model"].meanparam
            covariance = model["model"].covariance.__name__
            cov_params = model["model"].covparam
            initial_guess = model["parameters_initial_guess_procedure"]
            selection_criterion = model["pre_selection_criterion"]

            model_info += f"\nGaussian process {i}:\n"
            model_info += f"  Output Name: {model['output_name']}\n"
            model_info += f"  Mean: {mean_descr}\n"
            model_info += f"  Mean Type: {mean_type}\n"
            model_info += f"  Mean Parameters: {mean_params}\n"
            model_info += f"  Covariance: {covariance}\n"
            model_info += f"  Covariance Parameters: {cov_params}\n"
            model_info += f"  Initial Guess Procedure: {initial_guess.__name__ if initial_guess else 'None'}\n"
            model_info += f"  Selection Criterion: {selection_criterion.__name__ if selection_criterion else 'None'}\n"

        return model_info

    def set_mean_functions(self, mean_params):
        """Set mean functions.

        This function sets mean functions based on the parameters
        provided in mean_params.  Each entry in mean_params should
        specify a mean function and its associated parameters.

        Parameters
        ----------
        mean_params : list of dict or dict
            Parameters for defining mean functions. Can be a single
            dictionary applied to all outputs or a list of
            dictionaries, one for each output. Dictionary may include:
            - 'function': A callable mean function to be used
            - 'param_length': The length of the mean function's
              parameter vector, required if 'function' is given and is
              a callable.

        Returns
        -------
        tuple of list, list
            A tuple containing a list of mean function callables and a
            list of their descriptions.

        Raises
        ------
        ValueError
            If mean_params are not correctly specified or do not match
            output_dim.

        """
        if isinstance(mean_params, dict):
            mean_params = [mean_params] * self.output_dim
        elif not isinstance(mean_params, list) or len(mean_params) != self.output_dim:
            raise ValueError(
                "mean_params must be a dict or a list of dicts of length output_dim"
            )

        mean_functions = []
        mean_functions_info = []

        for i, param in enumerate(mean_params):
            if "function" in param and callable(param["function"]):
                mean_function = param["function"]
                if "param_length" not in param:
                    raise ValueError(
                        "'param_length' key is required for callable mean functions in mean_params"
                    )
                param_length = param["param_length"]
            else:
                mean_function, param_length = self.build_mean_function(i, param)

            mean_functions.append(mean_function)
            mean_functions_info.append(
                {"description": mean_function.__name__, "param_length": param_length}
            )

        return mean_functions, mean_functions_info

    def set_covariance_functions(self, covariance_params):
        """Set covariance functions.

        This method sets covariance functions based on the parameters
        provided in params. Each entry in params should specify a
        covariance function and its associated parameters.

        Parameters
        ----------
        covarariance_params : list of dict or dict
            Parameters for defining covariance functions. Can be a
            single dictionary applied to all outputs or a list of
            dictionaries, one for each output. Each dictionary may
            have a 'function' key providing a callable covariance
            function.

        Returns
        -------
        list
            A list of covariance function callables.

        Raises
        ------
        ValueError
            If params are not correctly specified or do not
            match output_dim.
        """
        if isinstance(covariance_params, dict):
            covariance_params = [covariance_params] * self.output_dim
        elif (
            not isinstance(covariance_params, list)
            or len(covariance_params) != self.output_dim
        ):
            raise ValueError(
                "params must be a dict or a list of dicts of length output_dim"
            )

        covariance_functions = []

        for i, param in enumerate(covariance_params):
            if "function" in param and callable(param["function"]):
                covariance_function = param["function"]
            else:
                covariance_function = self.build_covariance(i, param)

            covariance_functions.append(covariance_function)

        return covariance_functions

    def set_parameters_initial_guess_procedures(
        self, initial_guess_procedures=None, build_params=None
    ):
        """
        Set initial guess procedures based on provided initial_guess_procedures and parameters.

        Parameters
        ----------
        initial_guess_procedures : list or callable
            The initial guess procedures to be used.
        build_params : dict or list of dicts, optional
            Parameters for each initial guess procedure. Can be None.

        Returns
        -------
        list
            A list of initial guess procedure callables.

        Raises
        ------
        ValueError
            If the length of initial_guess_procedures or params does not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim

        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if initial_guess_procedures is None:
            initial_guess_procedures = [
                self.build_parameters_initial_guess_procedure(i, **(param or {}))
                for i, param in enumerate(build_params)
            ]
        elif (
            isinstance(initial_guess_procedures, list)
            and len(initial_guess_procedures) != self.output_dim
        ):
            raise ValueError(
                "initial_guess_procedures must be a list of length output_dim"
            )

        return initial_guess_procedures

    def set_selection_criteria(self, selection_criteria=None, build_params=None):
        """
        Set selection criteria based on provided selection_criteria and parameters.

        Parameters
        ----------
        selection_criteria : list or callable
            The selection criteria procedures to be used.
        build_params : dict or list of dicts, optional
            Parameters for each selection criterion. Can be None.

        Returns
        -------
        list
            A list of selection criterion callables.

        Raises
        ------
        ValueError
            If the length of selection_criteria or params does not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim

        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if selection_criteria is None:
            selection_criteria = [
                self.build_selection_criterion(i, **(param or {}))
                for i, param in enumerate(build_params)
            ]
        elif (
            isinstance(selection_criteria, list)
            and len(selection_criteria) != self.output_dim
        ):
            raise ValueError("selection_criteria must be a list of length output_dim")

        return selection_criteria

    def make_selection_criterion_with_gradient(
        self,
        model,
        xi_,
        zi_,
    ):
        pre_selection_criterion = model["pre_selection_criterion"]
        mean_paramlength = model["mean_paramlength"]

        if mean_paramlength > 0:
            # make a selection criterion with mean and covariance parameters
            def crit_(param):
                meanparam = param[:mean_paramlength]
                covparam = param[mean_paramlength:]
                l = pre_selection_criterion(
                    model["model"], meanparam, covparam, xi_, zi_
                )
                return l

        else:
            # make a selection criterion without mean parameter
            def crit_(covparam):
                l = pre_selection_criterion(model["model"], covparam, xi_, zi_)
                return l

        crit = gnp.jax.jit(crit_)
        dcrit = gnp.jax.jit(gnp.grad(crit))

        return crit, dcrit

    def select_params(self, xi, zi, force_param_initial_guess=True):
        """Parameter selection"""

        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        for i in range(self.output_dim):
            tic = time.time()

            model = self.models[i]
            mpl = model["mean_paramlength"]

            if model["model"].covparam is None or force_param_initial_guess:
                if mpl == 0:
                    meanparam0 = gnp.array([])
                    covparam0 = model["parameters_initial_guess_procedure"](
                        model["model"], xi_, zi_[:, i]
                    )
                else:
                    (meanparam0, covparam0) = model[
                        "parameters_initial_guess_procedure"
                    ](model["model"], xi_, zi_[:, i])
            else:
                meanparam0 = model["model"].meanparam
                covparam0 = model["model"].covparam

            param0 = gnp.concatenate((meanparam0, covparam0))

            crit, dcrit = self.make_selection_criterion_with_gradient(
                model, xi_, zi_[:, i]
            )

            # FIXME: Not in the spirit of the class.
            covparam_bounds = self.get_covparam_bounds(gnp.to_np(xi_), gnp.to_np(zi_[:, i]))

            meanparam_dim = meanparam0.shape[0]
            meanparam_bounds = [(-gnp.inf, gnp.inf)] * meanparam_dim
            bounds = meanparam_bounds + covparam_bounds

            param, info = gp.kernel.autoselect_parameters(
                param0, crit, dcrit, bounds=bounds, silent=True, info=True
            )

            assert not gnp.numpy.isnan(param).any()

            model["model"].meanparam = gnp.asarray(param[:mpl])
            model["model"].covparam = gnp.asarray(param[mpl:])
            model["info"] = info
            model["info"]["meanparam0"] = meanparam0
            model["info"]["covparam0"] = covparam0
            model["info"]["param0"] = param0
            model["info"]["meanparam"] = model["model"].meanparam
            model["info"]["covparam"] = model["model"].covparam
            model["info"]["param"] = param
            model["info"]["selection_criterion"] = crit
            model["info"]["time"] = time.time() - tic

    def predict(self, xi, zi, xt, convert_in=True, convert_out=True):
        """Predict method"""
        if zi.ndim == 1:
            zi = zi.reshape(-1, 1)

        zpm_ = gnp.empty((xt.shape[0], self.output_dim))
        zpv_ = gnp.empty((xt.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_predict = self.models[i]["model"].predict
            zpm_i, zpv_i = model_predict(
                xi, zi[:, i], xt, convert_in=convert_in, convert_out=False
            )
            zpm_ = gnp.set_col2(zpm_, i, zpm_i)
            zpv_ = gnp.set_col2(zpv_, i, zpv_i)

        if convert_out:
            zpm = gnp.to_np(zpm_)
            zpv = gnp.to_np(zpv_)
        else:
            zpm = zpm_
            zpv = zpv_

        return zpm, zpv

    def compute_conditional_simulations(
        self,
        xi,
        zi,
        xt,
        n_samplepaths=1,
        type="intersection",
        method="svd",
        convert_in=True,
        convert_out=True,
    ):
        """
        Generate conditional sample paths based on input data and simulation points.

        Parameters
        ----------
        xi : ndarray(ni, d)
            Input data points used in the GP model.
        zi : ndarray(ni, output_dim)
            Observations at the input data points xi.
        xt : ndarray(nt, d)
            Points at which to simulate.
        n_samplepaths : int, optional
            Number of sample paths to generate. Default is 1.
        type : str, optional
            Specifies the relationship between xi and xt. Can be 'intersection'
            (xi and xt may have a non-empty intersection) or 'disjoint'
            (xi and xt must be disjoint). Default is 'intersection'.
        method : str, optional
            Method to draw unconditional sample paths. Can be 'svd' or 'chol'. Default is 'svd'.

        Returns
        -------
        ndarray
            An array of conditional sample paths at simulation points xt.
            The shape of the array is (nt, n_samplepaths) for a single output model,
            and (nt, n_samplepaths, output_dim) for multi-output models.
        """
        xi_, zi_, xt_ = gp.core.Model.ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        compute_zsim = True  # FIXME: allows for reusing past computations
        if compute_zsim:
            # initialize xtsim and unconditional sample paths on xtsim
            ni = xi_.shape[0]
            nt = xt_.shape[0]

            xtsim = gnp.vstack((xi_, xt_))
            if type == "intersection":
                xtsim, indices = gnp.unique(xtsim, return_inverse=True, axis=0)
                xtsim_xi_ind = indices[0:ni]
                xtsim_xt_ind = indices[ni : (ni + nt)]
                n = xtsim.shape[0]
            elif type == "disjoint":
                xtsim_xi_ind = gnp.arange(ni)
                xtsim_xt_ind = gnp.arange(nt) + ni
                n = ni + nt

            # sample paths on xtsim
            zsim = gnp.empty((n, n_samplepaths, self.output_dim))

            for i in range(self.output_dim):
                zsim_i = self.models[i]["model"].sample_paths(
                    xtsim, n_samplepaths, method=method
                )
                zsim = gnp.set_col3(zsim, i, zsim_i)

        # conditional sample paths
        zpsim = gnp.empty((nt, n_samplepaths, self.output_dim))

        for i in range(self.output_dim):
            zpm, zpv, lambda_t = self.models[i]["model"].predict(
                xi_,
                zi_[:, i],
                xtsim[xtsim_xt_ind],
                return_lambdas=True,
                convert_in=False,
                convert_out=False,
            )

            if self.models[i]["model"].meantype == "linear_predictor":
                zpsim_i = self.models[i]["model"].conditional_sample_paths(
                    zsim[:, :, i],
                    xtsim_xi_ind,
                    zi_[:, i],
                    xtsim_xt_ind,
                    lambda_t,
                )
            elif self.models[i]["model"].meantype == "parameterized":
                zpsim_i = self.models[i][
                    "model"
                ].conditional_sample_paths_parameterized_mean(
                    zsim[:, :, i],
                    xi_,
                    xtsim_xi_ind,
                    zi_[:, i],
                    xt_,
                    xtsim_xt_ind,
                    lambda_t,
                )
            else:
                raise ValueError(
                    f"gpmp.core.Model.meantype {self.models[i]['model'].meantype} not implemented"
                )

            zpsim = gnp.set_col3(zpsim, i, zpsim_i)

        if self.output_dim == 1:
            # drop last dimension
            zpsim = zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))

        # r = {"xtsim": xtsim, "xtsim_xi_ind": xtsim_xi_ind, "xtsim_xt_ind": xtsim_xt_ind, "zsim": zsim}
        return zpsim

    def build_mean_function(self, output_idx: int, param: dict):
        """Build a mean function

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the mean function

        Returns
        -------
        (callable, int)
            The corresponding mean function and the number of parameters.

        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a covariance function

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the covariance function

        Returns
        -------
        callable
            A covariance function.

        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """Build an initial guess procedure for anisotropic parameters.

        Parameters
        ----------
        output_dim : int
            Number of output dimensions for the model.

        Returns
        -------
        function
            A function to compute initial guesses for anisotropic parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_selection_criterion(self, output_idx: int, **build_params):
        raise NotImplementedError("This method should be implemented by subclasses")


# ==============================================================================
# ModelMaternpREML Class
# ==============================================================================


class Model_MaternpREML(Model):
    def __init__(self, name, output_dim, mean_params, covariance_params):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        mean_params : dict or list of dicts
            Type of mean function to use.
        covariance_params : dict or list of dicts
            Parameters for each covariance function, including 'p'
        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=False,
            mean_params=mean_params,
            covariance_params=covariance_params,
        )

    def build_mean_function(self, output_idx: int, param: dict):
        """Build the mean function based on the mean type.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the mean function
            is being created.
        param : dict
            Must contain a "type" key with value "constant" or "linear".

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.

        """
        if "type" not in param:
            raise ValueError(f"Mean 'type' should be specified in 'param'")

        if param["type"] == "constant":
            return (mean_linpred_constant, 0)
        elif param["type"] == "linear":
            return (mean_linpred_linear, 0)
        else:
            raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a Matérn covariance function for a specific output
        index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the Matérn covariance function,
            including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.

        """
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError(
                f"Regularity 'p' should be integer and specified in 'param'"
            )

        p = param["p"]
        # FIXME: p = params.get("p", 2)  # Default value of p if not provided

        def maternp_covariance(x, y, covparam, pairwise=False):
            # Implementation of the Matérn covariance function using p and other parameters
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """Build an initial guess procedure for anisotropic parameters.

        Parameters
        ----------
        output_dim : int
            Number of output dimensions for the model.

        Returns
        -------
        function
            A function to compute initial guesses for anisotropic parameters.
        """

        def anisotropic_parameters_initial_guess(model, xi, zi):
            xi_ = gnp.asarray(xi)
            zi_ = gnp.asarray(zi).reshape(-1, 1)
            n = xi_.shape[0]
            d = xi_.shape[1]

            delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
            rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
            covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
            sigma2_GLS = 1.0 / n * model.norm_k_sqrd(xi_, zi_, covparam)

            return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))

        return anisotropic_parameters_initial_guess

    def build_selection_criterion(self, output_idx: int, **build_params):
        def reml_criterion(model, covparam, xi, zi):
            nlrel = model.negative_log_restricted_likelihood(covparam, xi, zi)
            return nlrel

        return reml_criterion


# ==============================================================================
# ModelMaternpML Class
# ==============================================================================


class Model_ConstantMeanMaternpML(Model):
    """GP model with a constant mean and a Matern covariance function. Parameters are estimated by ML"""

    def __init__(self, name, output_dim, rng, box, covariance_params=None):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        covariance_params : dict or list of dicts, optional
            Parameters for each covariance function, including 'p'
        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=True,
            mean_params={"type": "constant"},
            covariance_params=covariance_params,
            rng=rng,
            box=box
        )

    def build_mean_function(self, output_idx: int, param: dict):
        """Build the mean function based on the mean type.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Must contain a "type" key with value "constant".

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.

        """
        if "type" not in param:
            raise ValueError(f"Mean 'type' should be specified in 'param'")

        if param["type"] == "constant":
            return (mean_parameterized_constant, 1)
        else:
            raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a Matérn covariance function for a specific output
        index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        params : dict
            Additional parameters for the Matérn covariance function,
            including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.

        """
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError(
                f"Regularity 'p' should be integer and specified in 'param'"
            )

        p = param["p"]

        def maternp_covariance(x, y, covparam, pairwise=False):
            # Implementation of the Matérn covariance function using p and other parameters
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        return gp.kernel.anisotropic_parameters_initial_guess_constant_mean

    def build_selection_criterion(self, output_idx: int, **build_params):
        def ml_criterion(model, meanparam, covparam, xi, zi):
            nll = model.negative_log_likelihood(meanparam, covparam, xi, zi)
            return nll

        return ml_criterion

    def get_covparam_bounds(self, xi, zi):
        log_relative_amplitude = 60 * gnp.log(10)
        covparam_bounds = [
            (
                gnp.log(zi.var()) - log_relative_amplitude,
                gnp.log(zi.var()) + log_relative_amplitude
            )
        ]

        # FIXME: Calibrated for a Matérn covariance function with \nu = 5/2.
        delta_min = gnp.sqrt(xi.shape[1]) / 5
        delta_max = 10**(-5)
        for i in range(xi.shape[1]):
            dists = gnp.numpy.array([gnp.numpy.abs(xi[j, i] - xi[k, i]) for k in range(xi.shape[0]) for j in range(xi.shape[0])])
            min_dist = dists[dists > 0].min()
            max_dist = dists.max()

            upper_bound_min = -gnp.log(min_dist * delta_min)
            upper_bound_max = -gnp.log(max_dist * delta_max)

            upper_bound = min(upper_bound_min, upper_bound_max)

            covparam_bounds = covparam_bounds + [(-gnp.inf, upper_bound)]

        return covparam_bounds

# ==============================================================================
# ModelMaternp reGP Class
# ==============================================================================


class Model_ConstantMeanMaternp_reGP(Model_ConstantMeanMaternpML):
    """reGP model with a constant mean and a Matern covariance function."""

    def __init__(self, threshold_strategy_params, *args, crit_optim_options={}, **kwargs):
        """FIXME: comments"""

        super().__init__(*args, **kwargs)

        self.threshold_strategies, self.threshold_strategies_info = self.set_threshold_strategies(
            threshold_strategy_params
        )

        default_crit_optim_options = {"relaxed_init": "flat", "method": "SLSQP"}
        default_crit_optim_options.update(crit_optim_options)
        self.crit_optim_options = default_crit_optim_options


    def set_threshold_strategies(self, threshold_strategy_params):
        """FIXME: comments"""

        if isinstance(threshold_strategy_params, dict):
            threshold_strategy_params = [threshold_strategy_params] * self.output_dim
        elif not isinstance(threshold_strategy_params, list) or len(threshold_strategy_params) != self.output_dim:
            raise ValueError(
                "threshold_strategy_params must be a dict or a list of dicts of length output_dim"
            )

        threshold_strategies = []
        threshold_strategies_info = []

        for i, param in enumerate(threshold_strategy_params):
            if "function" in param and callable(param["function"]):
                threshold_strategy = param["function"]
            else:
                threshold_strategy = self.build_threshold_strategy(i, param)

            threshold_strategies.append(threshold_strategy)
            threshold_strategies_info.append(
                {"description": threshold_strategy.__name__}
            )

        return threshold_strategies, threshold_strategies_info

    def build_threshold_strategy(self, output_idx: int, param: dict):
        """FIXME: comments"""

        options = {}

        if "task" not in param:
            raise ValueError(f"'task' should be specified in 'param'")
        task = param["task"]

        if task == "levelset":
            if "t" not in param:
                raise ValueError(f"'t' should be specified in 'param' for task = 'levelset'")
            options["t"] = param["t"]

        if "strategy" not in param:
            raise ValueError(f"'strategy' should be specified in 'param'")
        strategy = param["strategy"]

        if "level" not in param:
            raise ValueError(f"'level' should be specified in 'param'")
        level = param["level"]

        if "n_ranges" in param:
            options["n_ranges"] = param["n_ranges"]
        else:
            # Default value
            options["n_ranges"]  = 10

        if strategy == "Constant":
            if "n_init" not in param:
                raise ValueError(f"'n_init' should be specified in 'param'")
            options["n_init"]  = param["n_init"]

        strategies_dict = getattr(regp, "{}_strategy".format(task))
        return strategies_dict[strategy](level, self.rng, self.box, options)

    # def build_mean_function(self, output_idx: int, param: dict):
    #     pass
    #
    # def build_covariance(self, output_idx: int, param: dict):
    #     pass

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        pass

    def build_selection_criterion(self, output_idx: int, **build_params):
        pass

    def compute_conditional_simulations(
        self,
        xi,
        zi,
        xt,
        n_samplepaths=1,
        type="intersection",
        method="svd",
        convert_in=True,
        convert_out=True,
    ):
        """
        Generate conditional sample paths based on input data and simulation points.

        Parameters
        ----------
        xi : ndarray(ni, d)
            Input data points used in the GP model.
        zi : ndarray(ni, output_dim)
            Observations at the input data points xi.
        xt : ndarray(nt, d)
            Points at which to simulate.
        n_samplepaths : int, optional
            Number of sample paths to generate. Default is 1.
        type : str, optional
            Specifies the relationship between xi and xt. Can be 'intersection'
            (xi and xt may have a non-empty intersection) or 'disjoint'
            (xi and xt must be disjoint). Default is 'intersection'.
        method : str, optional
            Method to draw unconditional sample paths. Can be 'svd' or 'chol'. Default is 'svd'.

        Returns
        -------
        ndarray
            An array of conditional sample paths at simulation points xt.
            The shape of the array is (nt, n_samplepaths) for a single output model,
            and (nt, n_samplepaths, output_dim) for multi-output models.
        """
        raise NotImplementedError

    def make_selection_criterion_with_gradient(
        self,
        model,
        xi_,
        zi_,
    ):
        raise NotImplementedError

    def select_params(self, xi, zi, force_param_initial_guess=True):
        """Parameter selection"""

        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        # Safer: one run with small length scales does not alter the subsequent ones.
        assert force_param_initial_guess

        self.zi_relaxed = gnp.copy(zi_)

        for i in range(self.output_dim):
            tic = time.time()

            model = self.models[i]
            mpl = model["mean_paramlength"]
            assert mpl == 1

            covparam_bounds = self.get_covparam_bounds(gnp.to_np(xi_), gnp.to_np(zi_[:, i]))

            G, R_list = self.threshold_strategies[i](gnp.to_np(xi_), gnp.to_np(zi_[:, i]))

            print("Build reGP model for G = {}".format(G))

            print("Select R")
            R = regp.select_optimal_R(
                model["model"],
                xi_,
                gnp.asarray(zi_[:, i]),
                G,
                R_list,
                covparam_bounds,
                optim_options=self.crit_optim_options,
            )

            print("Build model for selected R")
            self.models[i]["model"], self.zi_relaxed[:, i], _, info_ret = regp.remodel(
                model["model"],
                xi_,
                gnp.asarray(zi_[:, i]),
                R,
                covparam_bounds,
                True,
                optim_options=self.crit_optim_options,
            )
            print("reGP model built")

            self.models[i]["info"] = info_ret
            self.models[i]["R"] = R
            self.models[i]["param0"] = None
            self.models[i]["param"] = None
            self.models[i]["time"] = time.time() - tic

    def predict(self, xi, zi, xt, convert_in=True, convert_out=True):
        """Predict method"""
        assert self.zi_relaxed.ndim == 2

        zpm_ = gnp.empty((xt.shape[0], self.output_dim))
        zpv_ = gnp.empty((xt.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_predict = self.models[i]["model"].predict
            zpm_i, zpv_i = model_predict(
                xi, self.zi_relaxed[:, i], xt, convert_in=convert_in, convert_out=False
            )
            zpm_ = gnp.set_col2(zpm_, i, zpm_i)
            zpv_ = gnp.set_col2(zpv_, i, zpv_i)

        if convert_out:
            zpm = gnp.to_np(zpm_)
            zpv = gnp.to_np(zpv_)
        else:
            zpm = zpm_
            zpv = zpv_

        return zpm, zpv

# ==============================================================================
# Mean Functions Section
# ==============================================================================
# This section includes implementation of mean functions in GPmp


def mean_parameterized_constant(x, param):
    return param * gnp.ones((x.shape[0], 1))


def mean_linpred_constant(x, param):
    """Constant mean function for Gaussian Process models, linear predictor type.
    Parameters
    ----------
    x : ndarray(n, d)
        Input data points in dimension d.
    param : ndarray
        Parameters of the mean function (unused in constant mean).

    Returns
    -------
    ndarray
        Array of ones with shape (n, 1).
    """
    return gnp.ones((x.shape[0], 1))


def mean_linpred_linear(x, param):
    """Linear mean function for Gaussian Process models, linear predictor type.
    Parameters
    ----------
    x : ndarray(n, d)
        Input data points in dimension d.
    param : ndarray
        Parameters of the mean function (unused in linear mean).

    Returns
    -------
    ndarray
        Matrix [1, x_[1,1], ..., x_[1, d]
                1, x_[2,1], ..., x_[2, d]
                ...
                1, x_[n,1], ..., x_[n, d]]                   ]
    """
    return gnp.hstack((gnp.ones((x.shape[0], 1)), gnp.asarray(x)))
