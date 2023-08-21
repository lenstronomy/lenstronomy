__author__ = "aymgal"

import pytest
import os
import shutil
import numpy as np
import numpy.testing as npt

from lenstronomy.Sampling.Samplers.multinest_sampler import MultiNestSampler

try:
    import pymultinest
except:
    print(
        "Warning : MultiNest/pymultinest not installed properly, \
but tests will be trivially fulfilled"
    )
    pymultinest_installed = False
else:
    pymultinest_installed = True


_output_dir = "test_nested_out"


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood):
    """:param simple_einstein_ring_likelihood: fixture :return:"""
    likelihood, kwargs_truths = simple_einstein_ring_likelihood
    prior_means = likelihood.param.kwargs2args(**kwargs_truths)
    prior_sigmas = np.ones_like(prior_means) * 0.1
    sampler = MultiNestSampler(
        likelihood,
        prior_type="uniform",
        prior_means=prior_means,
        prior_sigmas=prior_sigmas,
        output_dir=_output_dir,
        remove_output_dir=True,
    )
    return sampler, likelihood


class TestMultiNestSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs_run = {
            "n_live_points": 10,
            "evidence_tolerance": 0.5,
            "sampling_efficiency": 0.8,  # 1 for posterior-only, 0 for evidence-only
            "importance_nested_sampling": False,
            "multimodal": True,
            "const_efficiency_mode": False,  # reduce sampling_efficiency to 5% when True
        }
        samples, means, logZ, logZ_err, logL, results = sampler.run(kwargs_run)
        assert len(means) == 1
        if not pymultinest_installed:
            # trivial test when pymultinest is not installed properly
            assert np.count_nonzero(samples) == 0
        if os.path.exists(_output_dir):
            shutil.rmtree(_output_dir, ignore_errors=True)

    def test_sampler_init(self, import_fixture):
        sampler, likelihood = import_fixture
        test_dir = "some_dir"
        os.mkdir(test_dir)
        sampler = MultiNestSampler(
            likelihood, prior_type="uniform", output_dir=test_dir
        )
        shutil.rmtree(test_dir, ignore_errors=True)
        try:
            sampler = MultiNestSampler(
                likelihood,
                prior_type="gaussian",
                prior_means=None,  # will raise an Error
                prior_sigmas=None,  # will raise an Error
                output_dir=None,
                remove_output_dir=True,
            )
        except Exception as e:
            assert isinstance(e, ValueError)
        try:
            sampler = MultiNestSampler(likelihood, prior_type="some_type")
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_prior(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        cube_low = np.zeros(n_dims)
        cube_upp = np.ones(n_dims)

        self.prior_type = "uniform"
        sampler.prior(cube_low, n_dims, n_dims)
        npt.assert_equal(cube_low, sampler.lowers)
        sampler.prior(cube_upp, n_dims, n_dims)
        npt.assert_equal(cube_upp, sampler.uppers)

        cube_mid = 0.5 * np.ones(n_dims)
        self.prior_type = "gaussian"
        sampler.prior(cube_mid, n_dims, n_dims)
        cube_gauss = np.array([1.0])
        npt.assert_equal(cube_mid, cube_gauss)

    def test_log_likelihood(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        args = np.nan * np.ones(n_dims)
        logL = sampler.log_likelihood(args, n_dims, n_dims)
        assert logL < 0
        # npt.assert_almost_equal(logL, -53.24465641401431, decimal=8)
        # assert logL == -1e15


if __name__ == "__main__":
    pytest.main()
