__author__ = "aymgal"

import pytest
import os
import shutil
import numpy as np
import numpy.testing as npt

from lenstronomy.Sampling.Samplers.polychord_sampler import DyPolyChordSampler

try:
    import dyPolyChord
except:
    print(
        "Warning : PolyChordLite/DyPolyChord not installed properly, \
but tests will be trivially fulfilled"
    )
    dypolychord_installed = False
else:
    dypolychord_installed = True

try:
    import nestcheck
except:
    print(
        "Warning : PolyChordLite/DyPolyChord not installed properly, \
but tests will be trivially fulfilled"
    )
    nestcheck_installed = False
else:
    nestcheck_installed = True

all_installed = dypolychord_installed and nestcheck_installed

_output_dir = "test_nested_out"


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood):
    """

    :param simple_einstein_ring_likelihood: fixture
    :return:
    """
    likelihood, kwargs_truths = simple_einstein_ring_likelihood
    prior_means = likelihood.param.kwargs2args(**kwargs_truths)
    prior_sigmas = np.ones_like(prior_means) * 0.1
    sampler = DyPolyChordSampler(
        likelihood,
        prior_type="uniform",
        prior_means=prior_means,
        prior_sigmas=prior_sigmas,
        output_dir=_output_dir,
        remove_output_dir=True,
    )
    return sampler, likelihood


class TestDyPolyChordSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs_run = {
            "ninit": 2,
            "nlive_const": 3,
        }
        dynamic_goal = 0.8
        samples, means, logZ, logZ_err, logL, results = sampler.run(
            dynamic_goal, kwargs_run
        )
        assert len(means) == 1
        if not all_installed:
            # trivial test when dypolychord is not installed properly
            assert np.count_nonzero(samples) == 0
        if os.path.exists(_output_dir):
            shutil.rmtree(_output_dir, ignore_errors=True)

    def test_sampler_init(self, import_fixture):
        sampler, likelihood = import_fixture
        test_dir = "some_dir"
        os.mkdir(test_dir)
        sampler = DyPolyChordSampler(
            likelihood, prior_type="uniform", output_dir=test_dir
        )
        shutil.rmtree(test_dir, ignore_errors=True)
        try:
            sampler = DyPolyChordSampler(
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
            sampler = DyPolyChordSampler(likelihood, prior_type="some_type")
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_prior(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        cube_low = np.zeros(n_dims)
        cube_upp = np.ones(n_dims)

        self.prior_type = "uniform"
        cube_low = sampler.prior(cube_low)
        npt.assert_equal(cube_low, sampler.lowers)
        cube_upp = sampler.prior(cube_upp)
        npt.assert_equal(cube_upp, sampler.uppers)

        cube_mid = 0.5 * np.ones(n_dims)
        self.prior_type = "gaussian"
        sampler.prior(cube_mid)
        cube_gauss = np.array([0.5])
        npt.assert_equal(cube_mid, cube_gauss)

    def test_log_likelihood(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        args = np.nan * np.ones(n_dims)
        logL, phi = sampler.log_likelihood(args)
        assert logL < 0
        # npt.assert_almost_equal(logL, -53.607122396369675, decimal=8)
        # assert logL == -1e15
        assert phi == []

    def test_write_equal_weights(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = 10
        ns_run = {
            "theta": np.zeros((1, n_dims)),
            "logl": np.zeros(1),
            "output": {
                "logZ": np.zeros(n_dims),
                "logZerr": np.zeros(n_dims),
                "param_means": np.zeros(n_dims),
            },
        }
        sampler._write_equal_weights(ns_run["theta"], ns_run["logl"])


if __name__ == "__main__":
    pytest.main()
