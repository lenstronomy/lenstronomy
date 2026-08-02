import copy
import sys
import types
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

    def test_sampler_init_mpi_branch(
        self, simple_einstein_ring_likelihood_2d, monkeypatch
    ):
        likelihood, _ = simple_einstein_ring_likelihood_2d

        class _FakeNautilusSamplerBackend(object):
            def __init__(self, prior, loglikelihood, n_dims, pool=None, **kwargs):
                self.loglikelihood = loglikelihood
                self.pool = pool
                self.kwargs = kwargs

        def _fake_check_install(self):
            self._nautilus = types.SimpleNamespace(Sampler=_FakeNautilusSamplerBackend)

        pool_kwargs = []

        class _FakeMPIPool(object):
            def __init__(self, **kwargs):
                pool_kwargs.append(kwargs)

            @staticmethod
            def is_master():
                return True

        nested_calls = []

        def _fake_set_nested(likelihood_module, n_dims):
            nested_calls.append((likelihood_module, n_dims))

        monkeypatch.setattr(NautilusSampler, "_check_install", _fake_check_install)
        fake_schwimmbad = types.SimpleNamespace(MPIPool=_FakeMPIPool)
        monkeypatch.setitem(sys.modules, "schwimmbad", fake_schwimmbad)
        monkeypatch.setattr(
            "lenstronomy.Sampling.Samplers.nautilus_sampler.set_nested_likelihood_module",
            _fake_set_nested,
        )

        sampler = NautilusSampler(likelihood_module=likelihood, mpi=True)

        assert pool_kwargs == [{}]
        assert len(nested_calls) == 1
        assert sampler._sampler.loglikelihood.__name__ == "nested_logl_worker"
        assert sampler._sampler.pool is not None


if __name__ == "__main__":
    pytest.main()
