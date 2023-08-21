import copy
import numpy as np
import pytest

from numpy.testing import assert_raises

from lenstronomy.Sampling.Samplers.nautilus_sampler import NautilusSampler


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood_2d):
    """:param simple_einstein_ring_likelihood_2d: fixture :return:"""
    likelihood, kwargs_truths = simple_einstein_ring_likelihood_2d
    sampler = NautilusSampler(likelihood_module=likelihood)
    return sampler, likelihood


class TestNautilusSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs = {
            "mpi": False,
            "verbose": True,
            "f_live": 1.0,
            "n_eff": 0.0,
            "seed": 42,
        }
        points, log_w, log_l, log_z = sampler.run(**kwargs)
        assert len(points) == len(log_w)
        assert len(points) == len(log_l)
        assert np.isfinite(log_z)


if __name__ == "__main__":
    pytest.main()
