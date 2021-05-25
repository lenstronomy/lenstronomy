"""
Created on Sep 30, 2013
modified on March 3-7, 2020

@authors: J. Akeret, S. Birrer, A. Shajib
"""

from copy import copy
from math import floor
import math
import numpy as np

__all__ = ['ParticleSwarmOptimizer']


class ParticleSwarmOptimizer(object):
    """
    Optimizer using a swarm of particles

    :param func:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param low: array of the lower bound of the parameter space
    :param high: array of the upper bound of the parameter space
    :param particle_count: the number of particles to use.
    :param threads: (optional)
        The number of threads to use for parallelization. If ``threads == 1``,
        then the ``multiprocessing`` module is not used but if
        ``threads > 1``, then a ``Pool`` object is created and calls to
        ``lnpostfn`` are run in parallel.

    :param pool: (optional)
        An alternative method of using the parallelized algorithm. If
        provided, the value of ``threads`` is ignored and the
        object provided by ``pool`` is used for all parallelization. It
        can be any object with a ``map`` method that follows the same
        calling sequence as the built-in ``map`` function.

    """

    def __init__(self, func, low, high, particle_count=25,
                 pool=None, args=None, kwargs=None):
        """

        :param func: function to call to return log likelihood
        :type func: python definition
        :param low: lower bound of the parameters
        :type low: numpy array
        :param high: upper bound of the parameters
        :type high: numpy array
        :param particle_count: number of particles in each iteration of the PSO
        :type particle_count: int
        :param pool: MPI pool for mapping different processes
        :type pool: None or MPI pool
        :param args: positional arguments to send to `func`. The function
        will be called as `func(x, *args, **kwargs)`.
        :type args: `list`
        :param kwargs: keyword arguments to send to `func`. The function
        will be called as `func(x, *args, **kwargs)`
        :type kwargs: `dict`
        """
        self.low = [l for l in low]
        self.high = [h for h in high]
        self.particleCount = particle_count
        self.pool = pool

        self.param_count = len(self.low)

        self.swarm = self._init_swarm()
        self.global_best = Particle.create(self.param_count)

        self.func = _FunctionWrapper(func, args, kwargs)

    def __getstate__(self):
        """
        In order to be generally pickleable, we need to discard the pool
        object before trying.
        """
        d = self.__dict__
        d["pool"] = None
        return d

    def __setstate__(self, state):
        self.__dict__ = state

    def set_global_best(self, position, velocity, fitness):
        """
        Set the global best particle.

        :param position: position of the new global best
        :type position: `list` or `ndarray`
        :param velocity: velocity of the new global best
        :type velocity: `list` or `ndarray`
        :param fitness: fitness of the new global best
        :type fitness: `float`
        :return: `None`
        :rtype:
        """
        self.global_best.position = [p for p in position]
        self.global_best.velocity = [v for v in velocity]
        self.global_best.fitness = fitness

    def _init_swarm(self):
        """
        Initiate the swarm.
        :return:
        :rtype:
        """
        swarm = []
        for _ in range(self.particleCount):
            swarm.append(
                Particle(np.random.uniform(self.low, self.high,
                                           size=self.param_count),
                         np.zeros(self.param_count)))

        return swarm

    def sample(self, max_iter=1000, c1=1.193, c2=1.193, p=0.7, m=1e-3, n=1e-2, early_stop_tolerance=None):
        """
        Launches the PSO. Yields the complete swarm per iteration

        :param max_iter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use
        :param m: stop criterion, difference between mean fitness and global
        best
        :param n: stop criterion, difference between norm of the particle
        vector and norm of the global best
        :param early_stop_tolerance: will terminate at the given value (should be specified as a chi^2)
        """

        self._get_fitness(self.swarm)
        i = 0
        while True:
            for particle in self.swarm:
                if self.global_best.fitness < particle.fitness:
                    self.global_best = particle.copy()
                    # if(self.isMaster()):
                    # print("new global best found %i %s"%(i,
                    # self.global_best.__str__()))

                if particle.fitness > particle.personal_best.fitness:
                    particle.update_personal_best()

            if i >= max_iter:
                if self.is_master():
                    print("Max iteration reached! Stopping.")
                return

            if self._converged(i, p=p, m=m, n=n):
                if self.is_master():
                    print("Converged after {} iterations!".format(i))
                    print("Best fit found: ", self.global_best.fitness,
                          self.global_best.position)
                return

            if early_stop_tolerance is not None:
                if self._acceptable_convergence(early_stop_tolerance):
                    return

            for particle in self.swarm:
                w = 0.5 + np.random.uniform(0, 1, size=self.param_count) / 2
                # w=0.72
                part_vel = w * np.array(particle.velocity)
                cog_vel = c1 * np.random.uniform(0, 1, size=self.param_count) \
                    * (np.array(particle.personal_best.position) -
                       np.array(particle.position))
                soc_vel = c2 * np.random.uniform(0, 1, size=self.param_count) \
                    * (np.array(self.global_best.position) -
                       np.array(particle.position))
                particle.velocity = (part_vel + cog_vel + soc_vel).tolist()
                particle.position = (np.array(particle.position) +
                                     np.array(particle.velocity)).tolist()

            self._get_fitness(self.swarm)

            swarm = []
            for particle in self.swarm:
                swarm.append(particle.copy())
            yield swarm

            i += 1

    def optimize(self, max_iter=1000, verbose=True, c1=1.193, c2=1.193,
                 p=0.7, m=1e-3, n=1e-2, early_stop_tolerance=None):
        """
        Run the optimization and return a full list of optimization outputs.

        :param max_iter: maximum iterations
        :param verbose: if `True`, print a message every 10 iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use
        :param m: stop criterion, difference between mean fitness and global
        best
        :param n: stop criterion, difference between norm of the particle
        vector and norm of the global best
        :param early_stop_tolerance: will terminate at the given value (should be specified as a chi^2)
        """
        chi2_list = []
        vel_list = []
        pos_list = []

        num_iter = 0
        for _ in self.sample(max_iter, c1, c2, p, m, n, early_stop_tolerance):
            chi2_list.append(self.global_best.fitness * 2)
            vel_list.append(self.global_best.velocity)
            pos_list.append(self.global_best.position)
            num_iter += 1

            if verbose and self.is_master():
                if num_iter % 10 == 0:
                    print(num_iter)

        return self.global_best.position, [chi2_list, pos_list, vel_list]

    def _get_fitness(self, swarm):
        """
        Set fitness (probability) of the particles in swarm.
        :param swarm: PSO state
        :type swarm: list of Particle() instances of the swarm
        :return:
        :rtype:
        """
        position = [particle.position for particle in swarm]
        if self.pool is None:
            map_func = map
        else:
            map_func = self.pool.map
        ln_probability = list(map_func(self.func, position))

        for i, particle in enumerate(swarm):
            particle.fitness = ln_probability[i]
            particle.position = position[i]

    def _converged(self, it, p, m, n):
        """
        Check for convergence.
        :param it:
        :type it:
        :param p:
        :type p:
        :param m:
        :type m:
        :param n:
        :type n:
        :return:
        :rtype:
        """
        #        test = self._converged_space2(p=p)
        #        print(test)
        fit = self._converged_fit(it=it, p=p, m=m)
        if fit:
            space = self._converged_space(it=it, p=p, m=n)
            return space
        else:
            return False

    def _converged_fit(self, it, p, m):
        """

        :param it:
        :type it:
        :param p:
        :type p:
        :param m:
        :type m:
        :return:
        :rtype:
        """
        best_sort = np.sort([particle.personal_best.fitness for particle in
                             self.swarm])[::-1]
        mean_fit = np.mean(best_sort[1:int(math.floor(self.particleCount * p))])
        #print( "best %f, mean_fit %f, ration %f"%( self.global_best[0],
        # mean_fit, abs((self.global_best[0]-mean_fit))))
        return abs(self.global_best.fitness - mean_fit) < m

    def _converged_space(self, it, p, m):
        """

        :param it:
        :type it:
        :param p:
        :type p:
        :param m:
        :type m:
        :return:
        :rtype:
        """
        sorted_swarm = [particle for particle in self.swarm]
        sorted_swarm.sort()
        best_of_best = sorted_swarm[0:int(floor(self.particleCount * p))]

        diffs = []
        for particle in best_of_best:
            diffs.append(np.array(self.global_best.position) -
                         np.array(particle.position))

        max_norm = max(list(map(np.linalg.norm, diffs)))
        return abs(max_norm) < m

    def _converged_space2(self, p):
        """

        :param p:
        :type p:
        :return:
        :rtype:
        """
        # Andres N. Ruiz et al.
        sorted_swarm = [particle for particle in self.swarm]
        sorted_swarm.sort()
        best_of_best = sorted_swarm[0:int(floor(self.particleCount * p))]

        positions = [particle.position for particle in best_of_best]
        means = np.mean(positions, axis=0)
        delta = np.mean((means - np.array(self.global_best.position)) /
                        np.array(self.global_best.position))
        return np.log10(delta) < -3.0

    def is_master(self):
        """
        Check if the current processor is the master.

        :return:
        :rtype:
        """
        if self.pool is None:
            return True
        else:
            return self.pool.is_master()

    def _acceptable_convergence(self, chi_square_tolerance):

        chi_square = -2 * self.global_best.fitness

        if np.min(chi_square) < chi_square_tolerance:
            return True
        else:
            return False


class Particle(object):
    """
    Implementation of a single particle

    :param position: the position of the particle in the parameter space
    :param velocity: the velocity of the particle
    :param fitness: the current fitness of the particle

    """
    def __init__(self, position, velocity, fitness=0):
        """

        :param position: parameter positions
        :param velocity: parameter velocity
        :param fitness:
        """
        self.position = [p for p in position]
        self.velocity = [v for v in velocity]

        self.fitness = fitness
        self.param_count = len(self.position)
        self._personal_best = None

    @property
    def personal_best(self):
        """

        :return:
        :rtype:
        """
        if self._personal_best is None:
            return self
        else:
            return self._personal_best

    @classmethod
    def create(cls, param_count):
        """
        Creates a new particle without position, velocity and -inf as fitness
        """

        return Particle(np.array([[]] * param_count),
                        np.array([[]] * param_count),
                        -np.Inf)

    def update_personal_best(self):
        """
        Sets the current particle representation as personal best
        """
        self._personal_best = self.copy()

    def copy(self):
        """
        Creates a copy of itself
        """
        return Particle(copy(self.position), copy(self.velocity), self.fitness)

    def __str__(self):
        """
        Get a `str` object for the particle state.
        :return:
        :rtype:
        """
        return "{:f}, pos: {:s} velocity: {:s}".format(self.fitness,
                                                       self.position,
                                                       self.velocity)

    def __lt__(self, other):
        return self.fitness > other.fitness

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __unicode__(self):
        return self.__str__()


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included. This hack is copied from
    emcee: https://github.com/dfm/emcee/.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:  # pragma: no cover
            import traceback

            print("PSO: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
