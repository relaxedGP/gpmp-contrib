# --------------------------------------------------------------
# Authors:
#   Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
#   Julien Bect <julien.bect@centralesupelec.fr>
# Copyright (c) 2023, 2024 CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import time
from numpy.random import default_rng
import scipy.stats as stats
import gpmp.num as gnp



class ParticlesSet:
    """
    A class representing a set of particles for Sequential Monte
    Carlo (SMC) simulation.

    This class provides elementary operations for initializing,
    reweighting, resampling, and moving particles.

    Parameters
    ----------
    box : array_like
        The domain box in which the particles are initialized.
    n : int, optional
        Number of particles (default is 1000).
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    n : int
        Number of particles.
    x : ndarray
        Current positions of the particles.
    logpx : ndarray
        Log-probabilities of the particles at their current positions.
    w : ndarray
        Weights of the particles.
    logpdf_function : callable
        Function to compute the log-probability density.
    param_s : float
        Scaling parameter for the perturbation step.
    rng : numpy.random.Generator
        Random number generator.

    Methods
    -------
    particles_init(box, n)
        Initialize particles within the given box.
    set_logpdf(logpdf_function)
        Set the log-probability density function.
    reweight()
        Reweight the particles based on the log-probability density function.
    ess()
        Calculate the effective sample size (ESS) of the particles.
    resample()
        Resample the particles based on their weights.
    perturb()
        Perturb the particles by adding random noise.
    move()
        Perform a Metropolis-Hastings step and compute the acceptation rate.

    """

    def __init__(self, box, n=1000, rng=default_rng()):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles
        self.dim = len(box[0])
        self.logpdf_function = None
        self.rng = rng

        # Initialize the particles.  Returns a tuple containing the
        # positions, log-probabilities, and weights of the particles
        self.x = None
        self.logpx = None
        self.w = None
        self.particles_init(box, n)

        # Metropolis-Hastings ingredients
        self.param_s = 0.05  # Default scaling parameter for perturbation

    def particles_init(self, box, n, method="randunif"):
        """Initialize particles within the given box.

        Parameters
        ----------
        box : array_like
            The domain box in which the particles are to be initialized.
        n : int
            Number of particles.
        method : str, optional
            Method for initializing particles. Currently, only
            'randunif' (uniform random) is supported. The option 'qmc'
            (quasi Monte-Carlo) will be supported in future versions.

        Returns
        -------
        tuple
            A tuple containing the positions, log-probabilities, and
            weights of the initialized particles.

        """
        assert self.dim == len(box[0]), "Box dimension do not match particle dimension."
        self.n = n

        # Initialize positions
        if method == "randunif":
            self.x = ParticlesSet.randunif(self.dim, self.n, box, self.rng)
        else:
            raise NotImplementedError(
                f"The method '{method}' is not supported. Currently, only 'randunif' is available."
            )

        # Initialize log-probabilities and weights
        self.logpx = gnp.zeros((n,))
        self.w = gnp.full((n,), 1 / n)

    def set_logpdf(self, logpdf_function):
        """
        Set the log-probability density function for the particles.

        Parameters
        ----------
        logpdf_function : callable
            Computes the log-probability density at given positions.
        """
        self.logpdf_function = logpdf_function

    def reweight(self):
        logpx_new = self.logpdf_function(self.x)
        self.w = self.w * gnp.exp(logpx_new - self.logpx)
        self.logpx = logpx_new

    def ess(self):
        """https://en.wikipedia.org/wiki/Effective_sample_size"""
        return gnp.sum(self.w) ** 2 / gnp.sum(self.w**2)

    def resample(self):
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        try:
            counts = self.multinomial_rvs(self.n, p, self.rng)
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
        self.w = gnp.full((self.n,), 1 / self.n)

    def perturb(self):
        assert self.param_s <= 10**4, "param_s is too high: {}".format(self.param_s)
        lower_bound_param_s = 10**(-12)
        if self.param_s < lower_bound_param_s:
            raise ParamSError(self.param_s, lower_bound_param_s, gnp.numpy.inf)

        C = self.param_s * gnp.cov(self.x.reshape(self.x.shape[0], -1).T)

        try:
            eps = ParticlesSet.multivariate_normal_rvs(C, self.n, self.rng)
        except ValueError as e:
            print("RW-MH move: non PSD matrix: \n{} ({})".format(C, e))
            base_relative_jitter = 10**(-16)
            n_auto_jitter = 17
            success = False
            for cpt in range(n_auto_jitter):
                relative_jitter = 10**(cpt) * base_relative_jitter
                C_jitter = C + relative_jitter * gnp.numpy.diag(C.diagonal())
                try:
                    eps = ParticlesSet.multivariate_normal_rvs(C_jitter, self.n, self.rng)
                    success = True
                    break
                except ValueError as inner_e:
                    print("RW-MH move: non PSD jittered matrix: \n{} ({})".format(C_jitter, inner_e))
                    pass
            if not success:
                raise RuntimeError("RW-MH move: non PSD jittered matrix: \n{}".format(C_jitter))

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        """
        Perform a Metropolis-Hastings step and compute the acceptation rate.

        This method perturbs the particles, computes the acceptation probabilities, and
        decides whether to move the particles to their new positions.

        Returns
        -------
        float
            Acceptation rate of the move.
        """
        # Perturb the particles
        y = self.perturb()
        logpy = self.logpdf_function(y)

        # Compute acceptation probabilities
        rho = gnp.minimum(1, gnp.exp(logpy - self.logpx))

        accepted_moves = 0    # Counter for accepted moves
        negligible_moves = 0  # Counter for 'negligible' moves, i.e., moves that do not change the particle
        for i in range(self.n):
            if ParticlesSet.rand(self.rng) < rho[i]:
                # Check if the RW move is numerically equal to the starting point.
                # The convention that the move is not accepted is used otherwise.
                if not (self.x[i, :] == y[i, :]).all():
                    accepted_moves += 1
                else:
                    negligible_moves += 1
                # Update the particle position and log probability if the move is accepted
                self.x = gnp.set_row2(self.x, i, y[i, :])
                self.logpx = gnp.set_elem1(self.logpx, i, logpy[i])

        # Warn about negligible moves
        if negligible_moves > 0:
            print("The RW proposed {} negligible moves.".format(negligible_moves))

        # Compute the acceptation rate
        acceptation_rate = accepted_moves / self.n

        return acceptation_rate

    @staticmethod
    def rand(rng):
        return rng.uniform()

    @staticmethod
    def multinomial_rvs(n, p, rng):
        return gnp.asarray(stats.multinomial.rvs(n=n, p=p, random_state=rng))

    @staticmethod
    def multivariate_normal_rvs(C, n, rng):
        return gnp.asarray(
            stats.multivariate_normal.rvs(cov=C, size=n, random_state=rng)
        )

    @staticmethod
    def randunif(dim, n, box, rng):
        return gnp.asarray(stats.qmc.scale(rng.uniform(size=(n, dim)), box[0], box[1]))


class SMC:
    """Sequential Monte Carlo (SMC) sampler class.

    This class drives the SMC process using a set of particles,
    employing a strategy as described in
    Bect, J., Li, L., & Vazquez, E. (2017). "Bayesian subset simulation",
    SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.
    Available at: https://arxiv.org/abs/1601.02557

    Parameters
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int, optional
        Number of particles (default is 1000).
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int
        Number of particles.
    particles : ParticlesSet
        Instance of ParticlesSet class to manage the particles.
    emergency_error_counter : int
        Counts the number of emergency_errors (used to raise a StoppingError).

    Methods
    -------
    step(logpdf_parameterized_function, logpdf_param)
        Perform a single SMC step.
    move_with_controlled_acceptation_rate()
        Adjust the particles' movement to control the acceptation rate.

    """

    def __init__(self, box, n=1000, mh_params={}, rng=default_rng()):
        """
        Initialize the SMC sampler.
        """
        self.box = box
        self.n = n
        self.particles = ParticlesSet(box, n, rng)

        # Dictionary to hold MH algorithm parameters
        self.mh_params = {
            "mh_steps": 5,
            "acceptation_rate_min": 0.4,
            "acceptation_rate_max": 0.6,
            "adjustment_factor": 1.4,
            "adjustment_max_iterations": 50,
        }

        self.mh_params.update(mh_params)

        self.emergency_error_counter = 0

        # Logging
        self.log = []  # Store the state logs
        self.stage = 0
        self.logging_current_ess = None
        self.logging_current_logpdf_param = None
        self.logging_target_logpdf_param = None
        self.logging_restart_iteration = 0
        self.logging_logpdf_param_sequence = []  # Sequence of logpdf_params in restart
        self.logging_acceptation_rate_sequence = []

    def step(self, logpdf_parameterized_function, logpdf_param):
        """
        Perform a single step of the SMC process.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density at
            given positions.
        logpdf_param: float
            Parameter value for the logpdf function (typically, a threshold).

        """
        # Set target density
        logpdf = lambda x: logpdf_parameterized_function(x, logpdf_param)
        self.particles.set_logpdf(logpdf)

        # Reweight
        self.particles.reweight()
        self.logging_current_ess = self.particles.ess()

        # Resample / move
        self.particles.resample()
        self.move_with_controlled_acceptation_rate()
        for _ in range(self.mh_params["mh_steps"] - 1):
            # Additional moves if required
            acceptation_rate = self.particles.move()
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

        # Logging
        self.logging_current_logpdf_param = logpdf_param
        self.log_state()

        # Check degeneracy
        min_prop = 0.2
        n_unique_particles = gnp.unique(self.particles.x, axis=0).shape[0]
        assert n_unique_particles >= min_prop * self.particles.x.shape[0], \
            "Too few unique particles: {} over {}".format(
            n_unique_particles,
            self.particles.x.shape[0],
        )

        # Debug plot, if needed
        debug = False
        if debug:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(self.particles.x, self.particles.logpx, ".")
            plt.show()

    def step_with_possible_restart(
        self,
        logpdf_parameterized_function,
        logpdf_initial_param,
        target_logpdf_param,
        min_ess_ratio,
        p0,
        debug=False,
    ):
        """Perform an SMC step with the possibility of restarting the process.

        This method checks if the effective sample size (ESS) falls
        below a specified ratio, and if so, initiates a restart. The
        restart process reinitializes particles and recalculates
        logpdf_params to better target the desired distribution.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density of a
            given position.
        logpdf_initial_param : float
            The starting logpdf_param value for the restart process.
        target_logpdf_param : float
            The desired target logpdf_param value for the log-probability
            density.
        min_ess_ratio : float
            The minimum acceptable ratio of ESS to the total number of
            particles. If the ESS falls below this ratio, a restart is
            initiated.
        p0 : float
            The prescribed probability used in the restart method to
            compute the new logpdf_param.
        debug : bool, optional
            If True, prints debug information during the
            process. Default is False.
        """
        # Logging
        self.stage += 1
        self.logging_current_logpdf_param = target_logpdf_param
        self.logging_target_logpdf_param = target_logpdf_param

        # Set target density
        logpdf = lambda x: logpdf_parameterized_function(x, target_logpdf_param)
        self.particles.set_logpdf(logpdf)

        # reweight
        self.particles.reweight()
        self.logging_current_ess = self.particles.ess()

        # restart?
        if self.logging_current_ess / self.n < min_ess_ratio:
            self.restart(
                logpdf_parameterized_function,
                logpdf_initial_param,
                target_logpdf_param,
                p0,
            )
            # Note: Logging will occur inside the restart method.
        else:
            # resample / move
            self.particles.resample()
            self.move_with_controlled_acceptation_rate()
            for _ in range(self.mh_params["mh_steps"] - 1):
                # Additional moves if required
                acceptation_rate = self.particles.move()
                self.logging_acceptation_rate_sequence.append(acceptation_rate)
            # Logging
            self.log_state()

    def restart(
        self,
        logpdf_parameterized_function,
        logpdf_initial_param,
        target_logpdf_param,
        p0,
        debug=False,
    ):
        """
        Perform a restart method in SMC.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric probability density
        logpdf_initial_param : float
            Starting param value.
        target_logpdf_param : float
            Target param value.
        p0 : float
            Prescribed probability
        debug : bool
            If True, print debug information.
        """
        if debug:
            print("---- Restarting SMC ----")

        self.log_state()  # Log current and target logpdf_params, ess

        self.particles.particles_init(self.box, self.n)
        current_logpdf_param = logpdf_initial_param

        self.logging_logpdf_param_sequence = [logpdf_initial_param]

        while current_logpdf_param != target_logpdf_param:
            next_logpdf_param = self.compute_next_logpdf_param(
                logpdf_parameterized_function,
                current_logpdf_param,
                target_logpdf_param,
                p0,
                debug,
            )
            self.logging_restart_iteration += 1
            self.logging_logpdf_param_sequence.append(next_logpdf_param)

            self.step(logpdf_parameterized_function, next_logpdf_param)

            current_logpdf_param = next_logpdf_param

        # Logging reinitialization
        self.logging_logpdf_param_sequence = []
        self.logging_restart_iteration = 0

    def subset(
        self,
        func,
        target,
        p0,
        xi,
        debug=False,
        max_iter=50
    ):
        """
        Perform a subset simulation.

        Parameters
        ----------
        func : callable
            The function on which to make the subset.
        target : float
            Target value.
        p0 : float
            Prescribed probability
        xi : array
            The current design-of-experiments.
        debug : bool
            If True, print debug information.
        debug : int
            Maximum number of steps to reach the target.
        """
        if debug:
            print("---- Start subset ----")

        # Subset-simulation message termination
        message = None

        emergency_error_counter_max = 5

        stopping_tol = 0.1
        max_criterion_xi = func(xi).numpy().max()

        diameter_crit_tol = 0.5

        self.particles.particles_init(self.box, self.n)
        u = - gnp.inf

        cpt = 0
        while u != target:
            criterion_particles = func(self.particles.x).numpy()
            next_u = min(target, gnp.numpy.quantile(criterion_particles, 1 - p0))
            assert u <= next_u <= target, (u, next_u, target)

            max_criterion_last_particles = criterion_particles.max()

            try:
                self.step(
                    lambda x, _u: gnp.log(func(x) >= _u),
                    next_u
                )
            except ParamSError as e:
                # FIXME: Define a global default value?
                self.particles.param_s = 0.05  # Default scaling parameter for perturbation
                if max_criterion_xi <= next_u:
                    message = str(e) + " Aborting only the subset-simulation since the best observation's value " \
                                  "has been reached in {} steps.".format(cpt)
                    break
                else:
                    self.emergency_error_counter += 1
                    if self.emergency_error_counter >= emergency_error_counter_max:
                        print(e)
                        raise StoppingError("Aborting the run since {} emergency errors have occurred successively"
                                            " without reaching the best observation's value.".format(
                                self.emergency_error_counter
                            )
                        )
                    else:
                        print(
                            e,
                            "Aborting only the subset-simulation since it is only the {}-th (max. {}) time"
                            "the best observation's value has not been reached in {}"
                            " steps.".format(self.emergency_error_counter, emergency_error_counter_max, cpt)
                        )
                        return
            u = next_u

            cpt += 1
            if cpt == max_iter:
                message = "Warning: maximum number of steps {} reached for subset-simulation." \
                          " Target: {}, Current: {}".format(max_iter, target, u)
                break

            if max_criterion_xi <= u:
                if (max_criterion_last_particles - u) <= stopping_tol * (max_criterion_last_particles - max_criterion_xi):
                    message = "Subset-simulation stopping after {} steps because the numerical tolerance is reached " \
                              "(max. design: {}, threshold: {}, max. particles: {}).".format(
                            cpt,
                            max_criterion_xi,
                            u,
                            max_criterion_last_particles
                    )
                    break

            diameter_particles = gnp.cdist_xx(self.particles.x).max()
            dist_xi_particles = gnp.custom_cdist(xi, self.particles.x).min()
            if diameter_particles <= diameter_crit_tol * dist_xi_particles:
                message = "Subset-simulation stopping after {} steps because the cloud has diameter {} " \
                          "which is less than a fraction {} of the distance {} to the design-of-experiments".format(
                        cpt,
                        diameter_particles,
                        diameter_crit_tol,
                        dist_xi_particles
                )
                break

        if message is None:
            message = "Subset-simulation performed in {} steps.".format(cpt)
        print(message)
        self.emergency_error_counter = 0

    def move_with_controlled_acceptation_rate(self, debug=False):
        """
        Adjust the particles' movement to maintain the acceptation
        rate within specified bounds.  This method dynamically adjusts
        the scaling parameter based on the acceptation rate to ensure
        efficient exploration of the state space.

        """
        iteration_counter = 0
        self.logging_acceptation_rate_sequence = []  # Logging
        while iteration_counter < self.mh_params["adjustment_max_iterations"]:
            iteration_counter += 1

            acceptation_rate = self.particles.move()

            # Logging
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

            if debug:
                print(f"Acceptation rate = {acceptation_rate}")

            if acceptation_rate < self.mh_params["acceptation_rate_min"]:
                self.particles.param_s /= self.mh_params["adjustment_factor"]
                continue

            if acceptation_rate > self.mh_params["acceptation_rate_max"]:
                self.particles.param_s *= self.mh_params["adjustment_factor"]
                continue

            break

    def _compute_p_value(self, logpdf_function, logpdf_param, logpdf_initial_param):
        """
        Compute the mean value of the exponentiated difference in
        log-probability densities between two logpdf_params.

        .. math::

            \\frac{1}{n} \\sum_{i=1}^{n} \\exp(logpdf_function(x_i, logpdf_param)
            - logpdf_function(x_i, logpdf_initial_param))

        Parameters
        ----------
        logpdf_function : callable
            Function to compute log-probability density.
        logpdf_param : float
            The current logpdf_param value.
        logpdf_initial_param : float
            The initial logpdf_param value used as a reference.

        Returns
        -------
        float
            Computed mean value.

        """
        return gnp.mean(
            gnp.exp(
                logpdf_function(self.particles.x, logpdf_param)
                - logpdf_function(self.particles.x, logpdf_initial_param)
            )
        )

    def compute_next_logpdf_param(
        self,
        logpdf_parameterized_function,
        logpdf_initial_param,
        target_logpdf_param,
        p0,
        debug=False,
    ):
        """
        Compute the next logpdf_param using a dichotomy method.

        This method is part of the restart strategy. It computes a
        logpdf_param for the parameter of the
        logpdf_parameterized_function, ensuring a controlled migration
        of particles to the next stage. The parameter p0 corresponds
        to the fraction of moved particles that will be in the support
        of the target density.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric log-probability density.
        logpdf_initial_param : float
            Starting logpdf_param value.
        target_logpdf_param : float
            Target logpdf_param value.
        p0 : float
            Prescribed probability.
        debug : bool
            If True, print debug information.

        Returns
        -------
        float
            Next computed logpdf_param.

        """
        tolerance = 0.05
        low = logpdf_initial_param
        high = target_logpdf_param

        # Check if target_logpdf_param can be reached with p >= p0
        p_target = self._compute_p_value(
            logpdf_parameterized_function, target_logpdf_param, logpdf_initial_param
        )
        if p_target >= p0:
            if debug:
                print("Target logpdf_param reached.")
            return target_logpdf_param

        while True:
            mid = (high + low) / 2
            p = self._compute_p_value(
                logpdf_parameterized_function, mid, logpdf_initial_param
            )

            if debug:
                print(
                    f"Search: p = {p}, "
                    + f"current logpdf_param / threshold = {mid}, "
                    + f"initial = {logpdf_initial_param}, "
                    + f"target = {target_logpdf_param}"
                )

            if abs(p - p0) < tolerance:
                break

            if p < p0:
                high = mid
            else:
                low = mid

        return mid

    def log_state(self):
        """
        Log the current state of the SMC process.
        """
        state = {
            "timestamp": time.time(),
            "stage": self.stage,
            "num_particles": self.n,
            "current_scaling_param": self.particles.param_s,
            "target_logpdf_param": self.logging_target_logpdf_param,
            "current_logpdf_param": self.logging_current_logpdf_param,
            "ess": self.logging_current_ess,
            "restart_iteration": self.logging_restart_iteration,
            "logpdf_param_sequence": self.logging_logpdf_param_sequence.copy(),
            "acceptation_rate_sequence": self.logging_acceptation_rate_sequence.copy(),
        }
        self.log.append(state)
        # reinitialize acceptation_rate_sequence
        self.logging_acceptation_rate_sequence = []

    def plot_state(self):
        """Plot the state of the SMC process over different stages.

        It includes visualizations of logpdf_params, effective sample
        size (ESS), and acceptation rates.
        """

        import matplotlib.pyplot as plt

        log_data = self.log

        def make_stairs(y):
            x_stairs = []
            y_stairs = []
            for i in range(len(y)):
                x_stairs.extend([i, i + 1])
                y_stairs.extend([y[i], y[i]])
            return x_stairs, y_stairs

        # Initializing lists to store data
        stages = []
        target_logpdf_params = []
        current_logpdf_params = []
        ess_values = []
        acceptation_rates = []
        stage_changes = []  # To mark the stages where change occurs

        # Extracting and replicating data according to the length of 'acceptation_rate_sequence' in each log entry
        for idx, entry in enumerate(log_data):
            ar_length = len(entry["acceptation_rate_sequence"])
            if ar_length == 0:
                entry["acceptation_rate_sequence"] = [0.0]
                ar_length = 1

            stages.extend([entry["stage"]] * ar_length)
            target_logpdf_params.extend([entry["target_logpdf_param"]] * ar_length)
            current_logpdf_params.extend([entry["current_logpdf_param"]] * ar_length)
            ess_values.extend([entry["ess"]] * ar_length)
            acceptation_rates.extend(entry["acceptation_rate_sequence"])

        # Plotting
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("logpdf_param", color=color)
        t, target_logpdf_params = make_stairs(target_logpdf_params)
        t, current_logpdf_params = make_stairs(current_logpdf_params)
        ax1.plot(
            t,
            target_logpdf_params,
            label="Target logpdf_param",
            color="red",
            linestyle="dashed",
        )
        ax1.plot(
            t,
            current_logpdf_params,
            label="Current logpdf_param",
            color="red",
            linestyle="solid",
        )
        (ymin, ymax) = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend(loc="upper left")

        # Adding vertical lines for stage changes
        last_stage = 0
        for idx, stage in enumerate(stages):
            if stage > last_stage:
                plt.axvline(x=idx, color="gray", linestyle="dashed")
                last_stage = stage

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("ESS", color=color)
        t, ess_values = make_stairs(ess_values)
        ax2.plot(t, ess_values, label="ESS", color=color)
        ax2.set_ylim(0.0, self.n)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = "tab:green"
        ax3.set_ylabel("Acceptation Rate", color=color)
        ax3.plot(
            acceptation_rates, label="Acceptation Rate", color=color, linestyle="dotted"
        )
        ax3.set_ylim(0.0, 1.0)
        ax3.tick_params(axis="y", labelcolor=color)
        ax3.legend(loc="lower right")

        fig.tight_layout()
        plt.title("SMC Process State Over Stages")
        plt.show()

# Error classes
class ParamSError(BaseException):
    def __init__(self, param_s, lower, upper):
        message = "ParamSError: param_s out of range (value: {}, lower bound: {}, upper_bound: {}).".format(
            param_s,
            lower,
            upper
        )
        super().__init__(message)

class StoppingError(BaseException):
    def __init__(self, sub_message):
        message = "Stopping error: {}".format(sub_message)
        super().__init__(message)
