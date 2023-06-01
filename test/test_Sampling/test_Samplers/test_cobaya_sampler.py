__author__ = 'nataliehogg'

import pytest
import numpy as np

from lenstronomy.Sampling.Samplers.cobaya_sampler import CobayaSampler

_outpath = 'cobaya_chain'

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

    def setup_method(self): # what is this for?
        pass

    def test_sampler(self, import_fixture):

        sampler, likelihood, means, sigmas = import_fixture

        num_params, param_names = likelihood.param.num_param()
        lower_limit, upper_limit = likelihood.param.param_limits()

        sampled_params = {k: {'prior': {'dist': 'uniform', 'min': lower_limit[i], 'max': upper_limit[i]}} for k, i in zip(param_names, range(len(param_names)))}

        [sampled_params[k].update({'ref': {'dist': 'norm', 'loc': means[i], 'scale': sigmas[i]}}) for k, i in zip(sampled_params.keys(), range(len(sampled_params)))]

        props = [0.001]*num_params

        [sampled_params[k].update({'proposal': props[i]}) for k, i in zip(sampled_params.keys(), range(len(props)))]

        info = {'likelihood': {'lenstronomy_likelihood': {'external': likelihood, 'input_params': sampled_params}}}

        info['params'] = sampled_params

        mcmc_kwargs = {'burn_in': 0,
                       'max_tries': 100*num_params,
                       'covmat': None,
                       'proposal_scale': 1,
                       'output_every': 500,
                       'learn_every': 40*num_params,
                       'learn_proposal': True,
                       'learn_proposal_Rminus1_max': 2,
                       'learn_proposal_Rminus1_max_early': 30,
                       'learn_proposal_Rminus1_min': 0,
                       'max_samples': np.inf,
                       'Rminus1_stop': 0.01,
                       'Rminus1_cl_stop': 0.2,
                       'Rminus1_cl_level': 0.95,
                       'Rminus1_single_split': 4,
                       'measure_speeds': True,
                       'oversample_power': 0.4,
                       'oversample_thin': True,
                       'drag': False}

        info['sampler'] = {'mcmc': mcmc_kwargs}

        info['output'] = _outpath

        info['force'] = True

        updated_info, sampler_name, best_fit_values = sampler.run(**info)

        # updated_info, sampler_name, best_fit_values = sampler.run()

        print(best_fit_values)

if __name__ == '__main__':
    pytest.main()
