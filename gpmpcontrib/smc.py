# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import scipy.stats as stats
import gpmp.num as gnp
import gpmp as gp


class ParticlesSet:
    """
    A class representing a set of particles for Sequential Monte Carlo (SMC) simulation.

    This class provides elementary operations for initializing, reweighting, resampling, and perturbing particles.

    Parameters
    ----------
    box : array_like
        The domain box in which the particles are initialized.
    n : int, optional
        Number of particles (default is 1000).

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

    Methods
    -------
    particles_init(box, n, method='randunif')
        Initialize particles within the given box.
    set_logpdf(logpdf_function)
        Set the log-probability density function.
    reweight()
        Reweight the particles based on the log-probability density function.
    ess()
        Calculate the effective sample size.
    resample()
        Resample the particles based on their weights.
    pertubate()
        Perturb the particles.
    move()
        Move the particles based on perturbation and acceptance ratio.
    """

    def __init__(self, box, n=1000):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles

        # Initialize the particles.  Returns a tuple containing the
        # positions, log-probabilities, and weights of the particles
        (self.x, self.logpx, self.w) = self.particles_init(box, n)

        self.logpdf_function = None

        # MH
        self.original_x = None  # Store the original positions
        self.original_logpx = None  # Store the original log probabilities
        self.param_s = 0.05  # Default scaling parameter for perturbation

    def particles_init(self, box, n, method='randunif'):
        """Initialize particles within the given box.

        Parameters
        ----------
        box : array_like
            The domain box in which the particles are to be initialized.
        n : int
            Number of particles.
        method : str, optional
            Method for initializing particles. Options are 'randunif' (uniform random) 
            or 'maximinlhs' (maximin Latin hypercube sampling).

        Returns
        -------
        tuple
            A tuple containing the positions, log-probabilities, and
            weights of the initialized particles.
        """
        dim = len(box[0])

        if method == 'randunif':
            x = gnp.asarray(gp.misc.designs.randunif(dim, n, box))
        elif method == 'maximinlhs':
            x = gnp.asarray(gp.misc.designs.maximinlhs(dim, n, box))

        logpx = gnp.zeros((n,))
        
        w = gnp.full((n, ), 1 / n)

        return (x, logpx, w)

    def set_logpdf(self, logpdf_function):
        """
        Set the log-probability density function for the particles.

        Parameters
        ----------
        logpdf_function : callable
            A function that computes the log-probability density at given positions.
        """
        self.logpdf_function = logpdf_function

    def reweight(self):
        logpx_new = self.logpdf_function(self.x)
        self.w = self.w * gnp.exp(logpx_new - self.logpx)
        self.logpx = logpx_new

    def ess(self):
        '''https://en.wikipedia.org/wiki/Effective_sample_size'''
        return gnp.sum(self.w)**2 / gnp.sum(self.w**2)

    def resample(self):
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        p = self.w / gnp.sum(self.w)
        try:
            counts = stats.multinomial.rvs(self.n, p)
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
        self.w = gnp.full((self.n, ), 1 / self.n)

    def pertubate(self):
        C = self.param_s * gnp.cov(self.x.reshape(self.x.shape[0], -1).T)
        eps = gnp.multivariate_normal.rvs(cov=C, n=self.n)

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        """
        Perform a Metropolis-Hastings step and compute the acceptance rate.

        This method perturbs the particles, computes the acceptance probabilities, and
        decides whether to move the particles to their new positions.

        Returns
        -------
        float
            The acceptance rate of the move.
        """
        # Store the original state to allow for canceling the move
        self.original_x = gnp.copy(self.x)
        self.original_logpx = gnp.copy(self.logpx)

        # Perturb the particles
        y = self.pertubate()
        logpy = self.logpdf_function(y)

        # Compute acceptance probabilities
        rho = gnp.minimum(1, gnp.exp(logpy - self.logpx))

        accepted_moves = 0  # Counter for accepted moves
        for i in range(self.n):
            if gnp.rand(1) < rho[i]:
                # Update the particle position and log probability if the move is accepted
                self.x = gnp.set_row2(self.x, i, y[i, :])
                self.logpx = gnp.set_elem1(self.logpx, i, logpy[i])
                accepted_moves += 1

        # Compute the acceptance rate
        acceptance_rate = accepted_moves / self.n

        # Debug plot, if needed
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(self.x, self.logpx, '.')
            plt.show()

        return acceptance_rate

    def cancel_move(self):
        """
        Cancel the last move and revert the particles to their original state.
        """
        if self.original_x is not None and self.original_logpx is not None:
            # Revert to the original state before the last move
            self.x = self.original_x
            self.logpx = self.original_logpx
        else:
            raise RuntimeError("No move to cancel or original state not stored")


class SMC:
    """
    Sequential Monte Carlo (SMC) sampler class.

    This class drives the SMC process using a set of particles.

    Parameters
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int, optional
        Number of particles (default is 1000).

    Attributes
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int
        Number of particles.
    particles : ParticlesSet
        Instance of ParticlesSet class to manage the particles.

    Methods
    -------
    step(logpdf)
        Perform a single SMC step.

    """
    def __init__(self, box, n=1000):
        """
        Initialize the SMC sampler.
        """
        self.box = box
        self.n = n
        self.particles = ParticlesSet(box, n)

    def move_with_controled_acceptance_rate(self):
        rho_min = 0.4
        rho_max = 0.6

        rho = self.particles.move()
        print(f"Acceptance rate = {rho}")
        
        if rho < rho_min:
            self.particles.param_s *= 0.9
            self.particles.cancel_move()
            self.move_with_controled_acceptance_rate()

        if rho > rho_max:
            self.particles.param_s *= 1.1
            self.particles.cancel_move()
            self.move_with_controled_acceptance_rate()

    def step(self, logpdf):
        """
        Perform a single step of the SMC process.

        Parameters
        ----------
        logpdf : callable
            A function that computes the log-probability density of a given position.
        """
        self.particles.set_logpdf(logpdf)
        self.particles.reweight()
        self.particles.resample()
        self.move_with_controled_acceptance_rate()
