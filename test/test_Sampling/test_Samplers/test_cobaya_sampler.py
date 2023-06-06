__author__ = 'nataliehogg'

import pytest
import numpy as np
from unittest import TestCase

from lenstronomy.Sampling.Samplers.cobaya_sampler import CobayaSampler

@pytest.fixture
def import_fixture(simple_einstein_ring_likelihood):
    """
    :param simple_einstein_ring_likelihood_2d: fixture
    :return:
    """
    likelihood, kwargs_truths = simple_einstein_ring_likelihood
    means = likelihood.param.kwargs2args(**kwargs_truths)
    sigmas = np.ones_like(means)*0.1
    sampler = CobayaSampler(likelihood_module=likelihood, mean_start=means, sigma_start=sigmas)
    return sampler, likelihood, means, sigmas

class TestCobayaSampler(object):
    """
    test cobaya
    """

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):

        sampler, likelihood, means, sigmas = import_fixture

        kwargs_cobaya = {'Rminus1_stop': 10}

        updated_info, sampler_name, best_fit_values = sampler.run(**kwargs_cobaya)

        assert str(sampler_name) == 'mcmc'

class TestRaise(TestCobayaSampler):

    def test_raise(self, import_fixture):

        sampler, likelihood, means, sigmas = import_fixture

        t = TestCase()

        with t.assertRaises(TypeError):
            # checks that TypeError is raised if prop widths not list or dict
            test_prop_type = {'proposal_widths': 0.1}
            sampler.run(**test_prop_type)

        with t.assertRaises(ValueError):
            # checks that ValueError is raised if wrong number of prop width
            test_prop_num = {'proposal_widths': [0.1, 0.1, 0.1]}
            sampler.run(**test_prop_num)

        with t.assertRaises(ValueError):
            # checks that ValueError is raised if wrong number of labels
            test_latex_num = {'latex': ['theta_{\rm E}', 'gamma']}
            sampler.run(**test_latex_num)

if __name__ == '__main__':
    pytest.main()
