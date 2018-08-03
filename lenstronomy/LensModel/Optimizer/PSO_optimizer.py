from __future__ import print_function, division, absolute_import, unicode_literals
from copy import copy
import numpy

"""
This class is adapted from the CosmoHammer Particle Swarm Optimizer routine.
"""


class ParticleSwarmOptimizer(object):

    '''
    Optimizer using a swarm of particles; adapted from CosmoHammer Particle Swarm Optimizer (different convergence criteria)

    '''


    def __init__(self, func, low, high, particleCount=25, verbose=False):
        '''
        Constructor
        '''
        self.func = func
        self.low = low
        self.high = high
        self.particleCount = particleCount
        self.verbose = verbose

        self.paramCount = len(self.low)
        self.gbest = Particle.create(self.paramCount)
        self.swarm = self._initSwarm()

    def _initSwarm(self):

        swarm = []
        for _ in range(self.particleCount):
            swarm.append(Particle(numpy.random.uniform(self.low, self.high, size=self.paramCount), numpy.zeros(self.paramCount)))

        return swarm

    def sample(self, maxIter=1000, c1=1.193, c2=1.193, lookback = 0.25, standard_dev = 0.01):
        """
        Launches the PSO. Yields the complete swarm per iteration

        :param maxIter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param lookback: percentange of particles to use when determining convergence
        :param standard_dev: standard deviation of the last lookback particles for convergence
        """
        self._get_fitness(self.swarm)
        i = 0
        self.i = i
        while True:

            for particle in self.swarm:
                if ((self.gbest.fitness)<particle.fitness):

                    self.gbest = particle.copy()

                if (particle.fitness > particle.pbest.fitness):
                    particle.updatePBest()

            if(i>=maxIter):
                if self.verbose:
                    print("max iteration reached! stoping")
                return

            if self.func.is_converged:
                return

            if self._converged_likelihood(maxIter*lookback,self.particleCount,standard_dev):
                return

            for particle in self.swarm:

                w = 0.5 + numpy.random.uniform(0,1,size=self.paramCount)/2
                #w=0.72
                part_vel = w * particle.velocity
                cog_vel = c1 * numpy.random.uniform(0,1,size=self.paramCount) * (particle.pbest.position - particle.position)
                soc_vel = c2 * numpy.random.uniform(0,1,size=self.paramCount) * (self.gbest.position - particle.position)
                particle.velocity = part_vel + cog_vel + soc_vel
                particle.position = particle.position + particle.velocity

            self._get_fitness(self.swarm)

            swarm = []
            for particle in self.swarm:
                swarm.append(particle.copy())
            yield swarm

            i+=1
            self.i = i

    def optimize(self, maxIter=1000, c1=1.193, c2=1.193, lookback=0.25, standard_dev=0.01):
        """
        Runs the complete optimiziation.

        :param maxIter: maximum iterations
        :param c1: cognitive weight
        :param c2: social weight
        :param p: stop criterion, percentage of particles to use
        :param m: stop criterion, difference between mean fitness and global best
        :param n: stop criterion, difference between norm of the particle vector and norm of the global best

        :return swarms, gBests: the swarms and the global bests of all iterations
        """

        #swarms = []
        gBests = []

        for swarm in self.sample(maxIter,c1,c2,lookback,standard_dev):

            #swarms.append(swarm)
            gBests.append(self.gbest.copy())
        print('done.')
        return gBests

    def _get_fitness(self,swarm):

        mapFunction = map

        pos = numpy.array([part.position for part in swarm])
        results =  mapFunction(self.func, pos)

        lnprob = numpy.array([l[0] for l in results])
        for i, particle in enumerate(swarm):
            particle.fitness = lnprob[i]

    def _converged_likelihood(self,min_i,look_back,standard_dev):

        """

        :param min_i: minimum number of iterations for convergence
        :param look_back: how many particles included in the convergence criterion
        :param standard_dev: the critical standard deviation of log-likelihood for convergence
        :return:
        """

        # don't return unless a certain number of iterations have happened
        if self.i < min_i:
            return False

        # compute the likelihood for the particles
        likelihood = [particle.fitness for particle in self.swarm]

        # compute the standard deviation of the last number (look_back) of particles

        return numpy.std(likelihood[-look_back:]) < standard_dev


class Particle(object):
    """
    Implementation of a single particle

    :param position: the position of the particle in the parameter space
    :param velocity: the velocity of the particle
    :param fitness: the current fitness of the particle

    """


    def __init__(self, position, velocity, fitness = 0):
        self.position = position
        self.velocity = velocity

        self.fitness = fitness
        self.paramCount = len(self.position)
        self.pbest = self

    @classmethod
    def create(cls, paramCount):
        """
        Creates a new particle without position, velocity and -inf as fitness
        """

        return Particle(numpy.array([[]]*paramCount),
                 numpy.array([[]]*paramCount),
                 -numpy.Inf)

    def updatePBest(self):
        """
        Sets the current particle representation as personal best
        """
        self.pbest = self.copy()

    def copy(self):
        """
        Creates a copy of itself
        """
        return Particle(copy(self.position),
                        copy(self.velocity),
                        self.fitness)

    def __str__(self):
        return "%f, pos: %s velo: %s"%(self.fitness, self.position, self.velocity)

    def __unicode__(self):
        return self.__str__()
