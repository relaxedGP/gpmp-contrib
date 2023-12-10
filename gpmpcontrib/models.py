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

# ==============================================================================
# Model Class
# ==============================================================================


class Model:
    def __init__(
        self,
        name,
        output_dim,
        parameterized_mean,
        mean_functions=None,
        mean_paramlengths=None,
        mean_names=None,
        covariance_functions=None,
        covariance_params=None,
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
            If True, the model uses a parameterized mean function. If False,
            it uses a linear predictor mean function.
        mean_functions : callable, list of callables, optional
            A list of mean functions, one for each output. If not provided,
            a default type is chosen based on the value of parameterized_mean.
        mean_paramlengths : int or list, optional
            Lengths of the mean parameter vector, to be used with mean_functions.
            Specifies the number of parameters for the mean functions.
        mean_names : str or list, optional
            Names or identifiers for the mean functions. Used if mean_functions
            are not explicitly provided.
        covariance_functions : list of callables, optional
            A list of covariance functions, one for each output.
        covariance_params : dict or list of dicts, optional
            Parameters for each covariance function, including 'p'
        initial_guess_procedures : list of callables, optional
            A list of procedures for initial guess of model parameters, one for each output.
        selection_criteria : list of callables, optional
            A list of selection criteria, one for each output.
        """
        self.name = name
        self.output_dim = output_dim

        self.parameterized_mean = parameterized_mean
        if self.parameterized_mean:
            mean_type = "parameterized"
        else:
            mean_type = "linear_predictor"
        self.mean_functions, self.mean_functions_info = self.set_mean_functions(
            mean_functions, mean_paramlengths, mean_names
        )
        self.covariance_functions = self.set_covariance_functions(
            covariance_functions, build_params=covariance_params
        )

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

    def set_mean_functions(
        self,
        mean_functions=None,
        mean_paramlengths=None,
        mean_names=None,
        build_params=None,
    ):
        """
        Set mean functions.

        This function uses provided mean_functions or calls the method
        self.build_mean_function with mean_names and build_params.

        Parameters
        ----------
        mean_functions : list or callable
            The mean functions to be used. Can be a single callable applied to all outputs
            or a list of callables, one for each output.
        mean_paramlengths : int, list of int, optional
            Lengths of the mean parameter vector, to be used with mean_functions.
            Specifies the number of parameters for the mean functions.
        mean_names : str or list
            Type of mean function to use. Can be "constant", "linear"... or a list of such strings.
        build_params : None
            Unused

        Returns
        -------
        tuple of list, list
            A tuple containing a list of mean function callables and a list of their names.

        Raises
        ------
        ValueError
            If mean_functions or mean_types are not correctly specified.
        NotImplementedError
            If a specified mean_type is not implemented.

        """
        if mean_functions is not None:
            if isinstance(mean_paramlength, int):
                mean_paramlengths = [mean_paramlengths] * self.output_dim
            elif (
                isinstance(mean_paramlengths, list)
                and len(mean_paramlengths) == self.output_dim
            ):
                pass
            else:
                raise ValueError(
                    "mean_paramlengths must be an int or a list of length output_dim"
                )
            if callable(mean_functions):
                mean_functions = [mean_functions for _ in range(self.output_dim)]
                mean_functions_info = [
                    {"description": "custom_function", "param_length": param_length}
                    for param_length in mean_paramlengths
                ]
            elif (
                isinstance(mean_functions, list)
                and len(mean_functions) == self.output_dim
            ):
                mean_functions_info = [
                    {
                        "description": "custom_function",
                        "param_length": mean_paramlengths[i],
                    }
                    for i in range(self.output_dim)
                ]
            else:
                raise ValueError(
                    "mean_functions must be a callable or a list of length output_dim"
                )
        elif isinstance(mean_names, list) and len(mean_names) == self.output_dim:
            mean_functions = []
            mean_functions_info = []
            for mn in mean_names:
                func, param_length = self.build_mean_function(mn)
                mean_functions.append(func)
                mean_functions_info.append(
                    {"description": mn, "param_length": param_length}
                )
        elif isinstance(mean_names, str):
            mean_functions = []
            mean_functions_info = []
            for _ in range(self.output_dim):
                func, param_length = self.build_mean_function(mean_names)
                mean_functions.append(func)
                mean_functions_info.append(
                    {"description": mean_names, "param_length": param_length}
                )
        else:
            raise ValueError("Specify mean_functions or mean_names")

        return mean_functions, mean_functions_info

    def set_covariance_functions(self, covariance_functions=None, build_params=None):
        """Set covariance functions

        This method uses provided covariance_functions or calls the method self.build_covariance

        Parameters
        ----------
        covariance_functions : list or callable, optional
            The covariance functions to be used.
        build_params : dict or list of dicts
            Parameters for each covariance function, including 'p'.

        Returns
        -------
        list
            A list of covariance function callables.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim

        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if covariance_functions is None:
            covariance_functions = []
            for i, param in enumerate(build_params):
                covariance_functions.append(self.build_covariance(i, **(param or {})))

        # If covariance_functions is already provided, validate its length
        elif (
            isinstance(covariance_functions, list)
            and len(covariance_functions) != self.output_dim
        ):
            raise ValueError("covariance_functions must be a list of length output_dim")

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
                l = pre_selection_criterion(model["model"], meanparam, covparam, xi_, zi_)
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
                    (meanparam0, covparam0) = model["parameters_initial_guess_procedure"](
                        model["model"], xi_, zi_[:, i]
                    )
            else:
                meanparam0 = model["model"].meanparam
                covparam0 = model["model"].covparam
                
            param0 = gnp.concatenate((meanparam0, covparam0))
                
            crit, dcrit = self.make_selection_criterion_with_gradient(
                model, xi_, zi_[:, i]
            )

            param, info = gp.kernel.autoselect_parameters(
                param0, crit, dcrit, silent=True, info=True
            )

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
                zpsim_i = self.models[i]["model"].conditional_sample_paths_parameterized_mean(
                    zsim[:, :, i],
                    xi_,
                    xtsim_xi_ind,
                    zi_[:, i],
                    xt_,
                    xtsim_xt_ind,
                    lambda_t,
                )
            else:
                raise ValueError(f"gpmp.core.Model.meantype {self.models[i]['model'].meantype} not implemented")
    
            zpsim = gnp.set_col3(zpsim, i, zpsim_i)

        if self.output_dim == 1:
            # drop last dimension
            zpsim = zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))

        # r = {"xtsim": xtsim, "xtsim_xi_ind": xtsim_xi_ind, "xtsim_xt_ind": xtsim_xt_ind, "zsim": zsim}
        return zpsim

    def build_mean_function(self, mean_name):
        """
        Build the mean function based on the mean type.

        Parameters
        ----------
        mean_name : str
            The type of mean function.

        Returns
        -------
        (callable, int)
            The corresponding mean function and the number of parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_covariance(self, output_idx: int, **params):
        """Create a covariance function

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function is being created.
        **params : dict
            Additional parameters for the covariance function

        Returns
        -------
        function
            A Matern covariance function.
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
    def __init__(self, name, output_dim, mean=None, covariance_params=None):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        mean : str or list, optional
            A default type of mean function to apply to outputs if mean_functions are not specified.
        covariance_params : dict or list of dicts, optional
            Parameters for each covariance function, including 'p'
        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=False,
            mean_names=mean,
            covariance_params=covariance_params,
        )

    def build_mean_function(self, mean_name):
        """
        Build the mean function based on the mean type.

        Parameters
        ----------
        mean_name : str
            The type of mean function.

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.
        """
        if mean_name == "constant":
            return (mean_linpred_constant, 0)
        elif mean_name == "linear":
            return (mean_linpred_linear, 0)
        else:
            raise NotImplementedError(f"Mean type {mean_name} not implemented")

    def build_covariance(self, output_idx: int, **params):
        """Create a Matérn covariance function for a specific output index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function is being created.
        **params : dict
            Additional parameters for the Matérn covariance function, including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.
        """
        p = params.get("p", 2)  # Default value of p if not provided

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
    def __init__(self, name, output_dim, covariance_params=None):
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
            mean_names="constant",
            covariance_params=covariance_params,
        )

    def build_mean_function(self, mean_name):
        """
        Build the mean function based on the mean type.

        Parameters
        ----------
        mean_name : str
            The type of mean function.

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.
        """
        if mean_name == "constant":
            return (mean_parameterized_constant, 1)
        else:
            raise NotImplementedError(f"Mean type {mean_name} not implemented")

    def build_covariance(self, output_idx: int, **params):
        """Create a Matérn covariance function for a specific output index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function is being created.
        **params : dict
            Additional parameters for the Matérn covariance function, including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.
        """
        p = params.get("p", 2)  # Default value of p if not provided

        def maternp_covariance(x, y, covparam, pairwise=False):
            # Implementation of the Matérn covariance function using p and other parameters
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        def anisotropic_parameters_initial_guess_constant_mean(model, xi, zi):
            """Anisotropic initialization strategy with a parameterized constant mean."""
            xi_ = gnp.asarray(xi)
            zi_ = gnp.asarray(zi).reshape((-1, 1))  # Ensure zi_ is a column vector
            n = xi_.shape[0]
            d = xi_.shape[1]

            delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
            rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

            covparam = gnp.concatenate((gnp.array([gnp.log(1.0)]), -gnp.log(rho)))
            zTKinvz, Kinv1, Kinvz = model.k_inverses(xi_, zi_, covparam)

            mean_GLS = gnp.sum(Kinvz) / gnp.sum(Kinv1)
            sigma2_GLS = (1.0 / n) * zTKinvz

            return mean_GLS.reshape(1), gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))

        return anisotropic_parameters_initial_guess_constant_mean

    def build_selection_criterion(self, output_idx: int, **build_params):
        def ml_criterion(model, meanparam, covparam, xi, zi):
            nll = model.negative_log_likelihood(meanparam, covparam, xi, zi)
            return nll

        return ml_criterion


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
