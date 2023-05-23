__author__ = 'nataliehogg'

# man with one sampling method always knows his posterior distribution; man with two never certain.

import numpy as np
from mpi4py import MPI # new requirement
from cobaya.run import run as crun
from cobaya.log import LoggedError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class CobayaSampler(object):

    def __init__(self, likelihood_module):
        """
        Pure Metropolis--Hastings MCMC sampling with Cobaya.

        If you use this sampler, you must cite the following works:

        Lewis & Bridle, https://arxiv.org/abs/astro-ph/0205436

        Lewis, https://arxiv.org/abs/1304.4473

        Torrado & Lewis, https://arxiv.org/abs/2005.05290 and https://ascl.net/1910.019

        For more information about Cobaya, see https://cobaya.readthedocs.io/en/latest/index.html

        :param likelihood_module: LikelihoodModule() instance

        """

        # get the logL and parameter info from LikelihoodModule
        self._ll = likelihood_module
        self._num_params, self._param_names = self._ll.param.num_param()
        self._lower_limit, self._upper_limit = self._ll.param.param_limits()

    def run(self, **kwargs):
        """
        docstring goes here
        """

        # get the parameters to be sampled
        sampled_params = self._param_names
        num_params = self._num_params

        # add the priors to the sampled_params
        sampled_params = {k: {'prior': {'min': self._lower_limit[i], 'max': self._upper_limit[i]}} for k, i in zip(sampled_params, range(len(sampled_params)))}

        # add reference values to start chain close to expected best fit
        # we could get these values internally from lenstronomy e.g. from _ll.param
        # that would mimic how the emcee implementation is done (with sampling_util/sample_ball_truncated())
        # but I like having it directly accessible/controllable by the user when they run the MCMC
        if 'starting_points' not in kwargs:
            pass
        else:
            refs = kwargs['starting_points']
            if len(refs) != len(sampled_params.keys()):
                raise ValueError('You must provide the same number of starting points as sampled parameters.')
            [sampled_params[k].update({'ref': refs[i]}) for k, i in zip(sampled_params.keys(), range(len(refs)))]

        # add proposal widths
        if 'proposal_widths' not in kwargs:
            pass
        else:
            props = kwargs['proposal_widths']
            if len(props) != len(sampled_params.keys()):
                raise ValueError('You must provide the same number of proposal widths as sampled parameters.')
            [sampled_params[k].update({'proposal': props[i]}) for k, i in zip(sampled_params.keys(), range(len(props)))]

        # add LaTeX labels so lenstronomy kwarg names don't break getdist plotting
        if 'latex' not in kwargs:
            print('No LaTeX labels provided. Manually edit the updated.yaml file to avoid lenstronomy parameter labels breaking GetDist.')
            pass
        else:
            latex = kwargs['latex']
            if len(latex) != len(sampled_params.keys()):
                raise ValueError('You must provide the same number of labels as sampled parameters.')
            [sampled_params[k].update({'latex': latex[i]}) for k, i in zip(sampled_params.keys(), range(len(latex)))]

        # likelihood function in cobaya-friendly format
        def likelihood_for_cobaya(**kwargs):
            current_input_values = [kwargs[p] for p in sampled_params]
            logp = self._ll.likelihood(current_input_values)
            return logp

        # gather all the information to pass to cobaya, starting with the likelihood
        info = {'likelihood': {'lenstronomy_likelihood': {'external': likelihood_for_cobaya, 'input_params': sampled_params}}}

        # for the above, can we do an args2kwargs for the 'output_params' key?? might bypass plotting issue

        # parameter info
        info['params'] = sampled_params

        # get all the kwargs for the mcmc sampler in cobaya
        # if not present, passes a default value (taken from cobaya docs)
        # note: parameter blocking kwargs not provided because fast/slow parameters are very case-by-case
        # also the temperature option is apparently deprecated

        mcmc_kwargs = {'burn_in': kwargs.get('burn_in', 0),
                       'max_tries': kwargs.get('max_tries', 40*num_params),
                       'covmat': kwargs.get('covmat', None),
                       'proposal_scale': kwargs.get('proposal_scale', 1),
                       'output_every': kwargs.get('output_every', 500),
                       'learn_every': kwargs.get('learn_every', 40*num_params),
                       'learn_proposal': kwargs.get('learn_proposal', True),
                       'learn_proposal_Rminus1_max': kwargs.get('learn_proposal_Rminus1_max', 2),
                       'learn_proposal_Rminus1_max_early': kwargs.get('learn_proposal_Rminus1_max_early', 30),
                       'learn_proposal_Rminus1_min': kwargs.get('learn_proposal_Rminus1_min', 0),
                       'max_samples': kwargs.get('max_samples', np.inf),
                       'Rminus1_stop': kwargs.get('Rminus1_stop', 0.01),
                       'Rminus1_cl_stop': kwargs.get('Rminus1_cl_stop', 0.2),
                       'Rminus1_cl_level': kwargs.get('Rminus1_cl_level', 0.95),
                       'Rminus1_single_split': kwargs.get('Rminus1_single_split', 4),
                       'measure_speeds': kwargs.get('measure_speeds', True),
                       'oversample_power': kwargs.get('oversample_power', 0.4),
                       'oversample_thin': kwargs.get('oversample_thin', True),
                       'drag': kwargs.get('drag', False)}

        # select mcmc as the sampler and pass the relevant kwargs
        info['sampler'] = {'mcmc': mcmc_kwargs}

        # where the chains and other files will be saved
        info['output'] = kwargs['path']

        # whether or not to overwrite previous chains with the same name (bool)
        info['force'] = kwargs['force_overwrite']

        # run the sampler
        # we wrap the call to crun to make sure any MPI exceptions are caught properly
        # this ensures the entire run will be terminated if any individual process breaks
        success = False
        try:
            updated_info, sampler = crun(info)
            success = True
        except LoggedError as err:
            pass
        success = all(comm.allgather(success))
        if not success and rank == 0:
            print('Sampling with MPI failed!')

        # get the best fit (max likelihood); format returned is a pandas series
        # this bypasses lenstronomy's way of doing it but matches lenstronomy result
        best_fit_series = sampler.collection.bestfit()

        # turn that pandas series into a list (of floats)
        best_fit_values = best_fit_series[sampled_params].values.tolist()

        return updated_info, sampler, best_fit_values
