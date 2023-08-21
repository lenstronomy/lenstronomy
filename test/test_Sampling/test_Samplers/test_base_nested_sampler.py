__author__ = 'aymgal'

import pytest
import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler


@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood):
    """

    :param simple_einstein_ring_likelihood: fixture
    :return:
    """
    likelihood, kwargs_truths = simple_einstein_ring_likelihood
    prior_means = likelihood.param.kwargs2args(**kwargs_truths)
    prior_sigmas = np.ones_like(prior_means) * 0.1
    sampler = NestedSampler(likelihood, 'gaussian', prior_means, prior_sigmas, 0.5, 0.5)
    return sampler, likelihood


class TestNestedSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):

        pass

    def test_sampler(self, import_fixture):
        sampler, likelihood = import_fixture
        kwargs_run = {}
        try:
            sampler.run(kwargs_run)
        except Exception as e:
            assert isinstance(e, NotImplementedError)

    def test_sampler_init(self, import_fixture):
        _, likelihood = import_fixture
        sampler = NestedSampler(likelihood, 'uniform', None, None, 1, 1)
        try:
            sampler = NestedSampler(likelihood, 'gaussian', None, None, 1, 1) # will raise an Error
        except Exception as e:
            assert isinstance(e, ValueError)
        try:
            sampler = NestedSampler(likelihood, 'some_type', None, None, 1, 1) # will raise an Error
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_prior(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        cube = np.zeros(n_dims)
        try:
            sampler.prior(cube)
        except Exception as e:
            assert isinstance(e, NotImplementedError)

    def test_log_likelihood(self, import_fixture):
        sampler, likelihood = import_fixture
        n_dims = sampler.n_dims
        args = np.nan * np.ones(n_dims)
        try:
            sampler.log_likelihood(args)
        except Exception as e:
            assert isinstance(e, NotImplementedError)


if __name__ == '__main__':
    pytest.main()
