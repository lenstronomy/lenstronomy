__author__ = 'aymgal'

import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Sampling.Samplers.dynesty_sampler import DynestySampler


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood):
    """

    :param simple_einstein_ring_likelihood: fixture
    :return:
    """
    likelihood, kwargs_truths = simple_einstein_ring_likelihood
    prior_means = likelihood.param.kwargs2args(**kwargs_truths)
    prior_means *= 1.01
    prior_sigmas = np.ones_like(prior_means)
    print(prior_sigmas, prior_means, 'test prior sigmas')
    sampler = DynestySampler(likelihood, prior_type='uniform',
                             prior_means=prior_means,
                             prior_sigmas=prior_sigmas,
                             sigma_scale=0.5)
    return sampler, likelihood


class TestDynestySampler(object):
    """
    test the fitting sequences
    """

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs_run = {
            'dlogz_init': 0.01,
            'nlive_init': 20,
            'nlive_batch': 20,
            'maxbatch': 1,
            'wt_kwargs': {'pfrac': 0.8},
        }
        samples, means, logZ, logZ_err, logL, results = sampler.run(kwargs_run)
        assert len(means) == 1

    def test_sampler_init(self, import_fixture):
        sampler, likelihood = import_fixture
        try:
            sampler = DynestySampler(likelihood, prior_type='gaussian',
                                     prior_means=None,  # will raise an Error
                                     prior_sigmas=None)  # will raise an Error
        except Exception as e:
            assert isinstance(e, ValueError)
        try:
            sampler = DynestySampler(likelihood, prior_type='some_type')
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_prior(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        cube_low = np.zeros(n_dims)
        cube_upp = np.ones(n_dims)

        self.prior_type = 'uniform'
        cube_low = sampler.prior(cube_low)
        npt.assert_equal(cube_low, sampler.lowers)
        cube_upp = sampler.prior(cube_upp)
        npt.assert_equal(cube_upp, sampler.uppers)

    def test_log_likelihood(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        args = np.nan * np.ones(n_dims)
        logL = sampler.log_likelihood(args)
        assert logL < 0
        # npt.assert_almost_equal(logL, -47.167446538898204)
        # assert logL == -1e15


if __name__ == '__main__':
    pytest.main()
