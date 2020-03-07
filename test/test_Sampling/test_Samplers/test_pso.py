"""
Test the PSO module.

Execute with py.test -v

"""
import numpy as np
import pytest
import time
import numpy.testing as npt

from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Sampling.Samplers.pso import Particle


class TestParticleSwarmOptimizer(object):
    """

    """
    ctx = None
    params = np.array([[1, 2, 3], [4, 5, 6]])

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_particle(self):
        """

        :return:
        :rtype:
        """
        particle = Particle.create(2)
        assert particle.fitness == -np.inf

        assert particle == particle.personal_best

        particle2 = particle.copy()
        assert particle.fitness == particle2.fitness
        assert particle.param_count == particle2.param_count
        assert np.all(particle.position == particle2.position)
        assert np.all(particle.velocity == particle2.velocity)

        particle.fitness = 1
        particle.update_personal_best()

        assert particle.personal_best.fitness == 1

    def test_setup(self):
        """

        :return:
        :rtype:
        """
        low = np.zeros(2)
        high = np.ones(2)
        pso = ParticleSwarmOptimizer(None, low, high, 10)

        assert pso.swarm is not None
        assert len(pso.swarm) == 10

        position = [part.position for part in pso.swarm]

        assert (position >= low).all()
        assert (position <= high).all()

        velocity = [part.velocity for part in pso.swarm]
        assert (velocity == np.zeros(2)).all()

        fitness = [part.fitness == 0 for part in pso.swarm]
        assert all(fitness)

        assert pso.global_best.fitness == -np.inf

    def test_optimize(self):
        """

        :return:
        :rtype:
        """
        low = np.zeros(2)
        high = np.ones(2)

        def func(p):
            return -np.random.rand()

        pso = ParticleSwarmOptimizer(func, low, high, 10)

        max_iter = 10
        swarms, global_bests = pso.optimize(max_iter)
        assert swarms is not None
        assert global_bests is not None
        assert len(swarms) == max_iter
        assert len(global_bests) == max_iter

        fitness = [part.fitness != 0 for part in pso.swarm]
        assert all(fitness)

        assert pso.global_best.fitness != -np.inf

    def test_sample(self):
        """

        :return:
        :rtype:
        """
        np.random.seed(42)
        n_particle = 100
        n_iterations = 100

        def ln_probability(x):
            return - x**2

        pso = ParticleSwarmOptimizer(func=ln_probability, low=[-10], high=[10],
                                     particle_count=n_particle, threads=1)

        init_pos = np.array([1])
        pso.global_best.position = init_pos
        pso.global_best.velocity = [0] * len(init_pos)
        pso.global_best.fitness = ln_probability(init_pos)
        x2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()

        if pso.is_master():
            print('Computing the PSO...')

        num_iter = 0

        for swarm in pso.sample(n_iterations):
            x2_list.append(pso.global_best.fitness * 2)
            vel_list.append(pso.global_best.velocity)
            pos_list.append(pso.global_best.position)
            num_iter += 1
            if pso.is_master():
                if num_iter % 10 == 0:
                    print(num_iter)

        result = pso.global_best.position

        time_end = time.time()
        print(time_end - time_start)
        print(result)
        npt.assert_almost_equal(result[0], 0, decimal=6)


if __name__ == '__main__':
    pytest.main()
