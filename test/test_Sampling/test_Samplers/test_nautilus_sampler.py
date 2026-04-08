import copy
import numpy as np
import pytest

from numpy.testing import assert_raises

from lenstronomy.Sampling.Samplers.nautilus_sampler import NautilusSampler


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood_2d):
    """

    :param simple_einstein_ring_likelihood_2d: fixture
    :return:
    """
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
        samples, means, log_z, log_z_err, log_l, results = sampler.run(**kwargs)
        assert len(samples) == len(log_l)
        assert len(means) == samples.shape[1]
        assert np.isfinite(log_z)
        assert np.isfinite(log_z_err)
        assert "points" in results
        assert "log_w" in results
        assert "log_l" in results


if __name__ == "__main__":
    pytest.main()
