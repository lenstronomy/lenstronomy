"""
Created on Sep 30, 2013
modified on Mars 3, 2020

@author: J. Akeret, S. Birrer, A. Shajib
"""

from copy import copy
from math import floor
import math
import numpy as np
from schwimmbad import SerialPool


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

    def __init__(self, func, low, high, particle_count=25, threads=1,
                 pool=None):
        """
        Constructor
        """
        self.func = func
        self.low = low
        self.high = high
        self.particleCount = particle_count
        self.threads = threads
        self.pool = pool

        if self.pool is None:
            self.pool = SerialPool()  # this uses default map() in python

        self.param_count = len(self.low)

        self.swarm = self._init_swarm()
        self.global_best = Particle.create(self.param_count)

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
        self.global_best.position = position
        self.global_best.velocity = velocity
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

    def sample(self, max_iter=1000, c1=1.193, c2=1.193, p=0.7, m=10 ** -3,
               n=10 ** -2):
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
                print("Max iteration reached! Stopping.")
                return

            if self._converged(i, p=p, m=m, n=n):
                if self.is_master():
                    print("Converged after {} iterations!".format(i))
                    print("Best fit found: ", self.global_best.fitness,
                          self.global_best.position)
                return

            for particle in self.swarm:
                w = 0.5 + np.random.uniform(0, 1, size=self.param_count) / 2
                # w=0.72
                part_vel = w * particle.velocity
                cog_vel = c1 * np.random.uniform(0, 1, size=self.param_count) \
                    * (particle.personal_best.position - particle.position)
                soc_vel = c2 * np.random.uniform(0, 1, size=self.param_count) \
                    * (self.global_best.position - particle.position)
                particle.velocity = part_vel + cog_vel + soc_vel
                particle.position = particle.position + particle.velocity

            self._get_fitness(self.swarm)

            swarm = []
            for particle in self.swarm:
                swarm.append(particle.copy())
            yield swarm

            i += 1

    def optimize(self, max_iter=1000, c1=1.193, c2=1.193, p=0.7, m=10 ** -3, n=10 ** -2):
        """
        Runs the complete optimization.

        :param max_iter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use
        :param m: stop criterion, difference between mean fitness and global
        best
        :param n: stop criterion, difference between norm of the particle
        vector and norm of the global best

        :return swarms, global_bests: the swarms and the global bests of all
        iterations
        """

        swarms = []
        global_bests = []
        for swarm in self.sample(max_iter, c1, c2, p, m, n):
            swarms.append(swarm)
            global_bests.append(self.global_best.copy())

        return swarms, global_bests

    def _get_fitness(self, swarm):
        """
        Get fitness (probability) of the particles.
        :param swarm:
        :type swarm:
        :return:
        :rtype:
        """
        position = np.array([part.position for part in swarm])
        ln_probability = list(self.pool.map(self.func, position))

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
        mean_fit = np.mean(best_sort[1:math.floor(self.particleCount * p)])
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
        sorted_swarm.sort(key=lambda part: -part.fitness)
        best_of_best = sorted_swarm[0:int(floor(self.particleCount * p))]

        diffs = []
        for particle in best_of_best:
            diffs.append(self.global_best.position - particle.position)

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
        sorted_swarm.sort(key=lambda part: -part.fitness)
        best_of_best = sorted_swarm[0:int(floor(self.particleCount * p))]

        positions = [particle.position for particle in best_of_best]
        means = np.mean(positions, axis=0)
        delta = np.mean((means - self.global_best.position) /
                        self.global_best.position)
        return np.log10(delta) < -3.0

    def is_master(self):
        """

        :return:
        :rtype:
        """
        return self.pool.is_master()


class Particle(object):
    """
    Implementation of a single particle

    :param position: the position of the particle in the parameter space
    :param velocity: the velocity of the particle
    :param fitness: the current fitness of the particle

    """
    def __init__(self, position, velocity, fitness=0):
        self.position = position
        self.velocity = velocity

        self.fitness = fitness
        self.param_count = len(self.position)
        self.personal_best = self

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
        self.personal_best = self.copy()

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

    def __unicode__(self):
        return self.__str__()
