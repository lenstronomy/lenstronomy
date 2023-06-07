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
    test cobaya; we have separate functions to pass through all the different if/else options
    thus maximising test coverage
    """

    def setup_method(self):
        pass

    def test_sampler(self, import_fixture):
        '''
        function to test the basic sampler
        '''

        sampler, likelihood, means, sigmas = import_fixture

        test_cobaya = {'Rminus1_stop': 100}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_cobaya)

        assert str(sampler_name) == 'mcmc'

    def test_labels(self, import_fixture):
        '''
        function to test that latex labels are correctly read
        '''

        sampler, likelihood, means, sigmas = import_fixture

        test_labels_kwargs = {'Rminus1_stop': 100, 'labels': ['theta_{\rm E}']}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_labels_kwargs)

        assert str(sampler_name) == 'mcmc'

    def test_props(self, import_fixture):
        '''
        function to test passing a dict for the proposal widths
        '''

        sampler, likelihood, means, sigmas = import_fixture

        props = {'theta_E': 0.001}

        test_prop_kwargs = {'Rminus1_stop': 100, 'proposal_widths': props}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_prop_kwargs)

        assert str(sampler_name) == 'mcmc'

    def test_mpi(self, import_fixture):
        '''
        function to test mpi option
        '''
        sampler, likelihood, means, sigmas = import_fixture

        test_mpi_kwargs = {'Rminus1_stop': 100, 'mpi': True}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_mpi_kwargs)

        assert str(sampler_name) == 'mcmc'

    def test_path(self, import_fixture):
        '''
        function to test passing an outpath
        '''

        sampler, likelihood, means, sigmas = import_fixture

        test_path_kwargs = {'Rminus1_stop': 100, 'output': 'test_chain'}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_path_kwargs)

        assert str(sampler_name) == 'mcmc'

class TestRaise(TestCobayaSampler):
    # is it ok to use unittest rather than pyttest here?
    # it seems a bit clearer

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
