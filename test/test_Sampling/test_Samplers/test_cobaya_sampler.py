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
        '''
        function to test the sampler
        '''
        # test the sampler
        sampler, likelihood, means, sigmas = import_fixture

        test_cobaya = {'Rminus1_stop': 100}

        updated_info, sampler_name, best_fit_values = sampler.run(**test_cobaya)

        assert str(sampler_name) == 'mcmc'

        # test labels
        sampler, likelihood, means, sigmas = import_fixture

        test_labels_kwargs = {'Rminus1_stop': 100, 'latex': ['theta_{\rm E}']}

        updated_info_l, sampler_name_l, best_fit_values_l = sampler.run(**test_labels_kwargs)

        assert str(sampler_name_l) == 'mcmc'

        # test passing dict for proposals
        sampler, likelihood, means, sigmas = import_fixture

        props = {'theta_E': 0.001}

        test_prop_kwargs = {'Rminus1_stop': 100, 'proposal_widths': props}

        updated_info_d, sampler_name_d, best_fit_values_d = sampler.run(**test_prop_kwargs)

        assert str(sampler_name_d) == 'mcmc'

        # test passing path
        sampler, likelihood, means, sigmas = import_fixture

        test_path_kwargs = {'Rminus1_stop': 100, 'path': 'test_chain'}

        updated_info_p, sampler_name_p, best_fit_values_p = sampler.run(**test_path_kwargs)

        assert str(sampler_name_p) == 'mcmc'

        # use unittest to test raised exceptions
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

        with t.assertRaises(ValueError):
            # checks that ValueError is raised if drag is passed
            test_drag = {'drag': True}
            sampler.run(**test_drag)


if __name__ == '__main__':
    pytest.main()
