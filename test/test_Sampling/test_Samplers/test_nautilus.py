import copy

import pytest
from numpy.testing import assert_raises

from lenstronomy.Sampling.Samplers.nautilus import Nautilus


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood_2d):
    """

    :param simple_einstein_ring_likelihood_2d: fixture
    :return:
    """
    likelihood, kwargs_truths = simple_einstein_ring_likelihood_2d
    sampler = Nautilus(likelihood_module=likelihood)
    return sampler, likelihood


class TestNautilusSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs_run = {
            "prior_type": "uniform",
            "mpi": False,
            "thread_count": 1,
            "verbose": True,
            "one_step": True,
            "n_live": 2,
            "random_state": 42,
        }
        points, log_w, log_l, log_z = sampler.nautilus_sampling(**kwargs_run)
        assert len(points) == 100
        assert len(log_l) == 100

        kwargs_run_fail = copy.deepcopy(kwargs_run)
        kwargs_run_fail["prior_type"] = "wrong"
        assert_raises(ValueError, sampler.nautilus_sampling, **kwargs_run_fail)

    def test_prior(self):
        num_param = 10
        from nautilus import Prior

        prior = Prior()

        for i in range(num_param):
            prior.add_parameter(dist=(0, 1))
        assert num_param == prior.dimensionality()


if __name__ == "__main__":
    pytest.main()
