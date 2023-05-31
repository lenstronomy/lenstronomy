__author__ = 'nataliehogg'

# man with one sampling method always knows his posterior distribution; man with two never certain.

import numpy as np
from mpi4py import MPI
from cobaya.run import run as crun
from cobaya.log import LoggedError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class CobayaSampler(object):

    def __init__(self, likelihood_module, mean_start, sigma_start):
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

        self._mean_start = mean_start
        self._sigma_start = sigma_start

    def run(self, **kwargs):
        """
        docstring goes here
        """

        sampled_params = self._param_names

        # add the priors to the sampled_params
        # currently a uniform prior is hardcoded for all params
        # cobaya allows any 1D continuous dist in scipy.stats; thinking how to implement this here
        sampled_params = {k: {'prior': {'dist': 'uniform', 'min': self._lower_limit[i], 'max': self._upper_limit[i]}} for k, i in zip(sampled_params, range(len(sampled_params)))}

        # add reference values to start chain close to expected best fit
        # this hardcodes a Gaussian and uses the sigma_kwargs passed by the user
        # again cobaya allows any 1D continous distribution; thinking how to implement this
        # tricky with current info internal in lenstronomy
        [sampled_params[k].update({'ref': {'dist': 'norm', 'loc': self._mean_start[i], 'scale': self._sigma_start[i]}}) for k, i in zip(sampled_params.keys(), range(len(sampled_params)))]

        # add proposal widths
        # first check if proposal_widths has been passed
        if 'proposal_widths' not in kwargs:
            pass
        else:
            # check if what's been passed is dict
            if isinstance(kwargs['proposal_widths'], dict):
                # if yes, convert to list
                props = list(kwargs['proposal_widths'].values())
            elif isinstance(kwargs['proposal_widths'], list):
                # if no and it's a list, do nothing
                props = kwargs['proposal_widths']
            else:
                # if no and not a list, raise TypeError
                raise TypeError('Proposal widths must be a list of floats or a dictionary of parameters and floats.')
            # check the right number of values are present
            if len(props) != len(sampled_params.keys()):
                # if not, raise ValueError
                raise ValueError('You must provide the same number of proposal widths as sampled parameters.')
            # update sampled_params dict with proposal widths
            [sampled_params[k].update({'proposal': props[i]}) for k, i in zip(sampled_params.keys(), range(len(props)))]

        # add LaTeX labels so lenstronomy kwarg names don't break getdist plotting
        # first check if the labels have been passed
        if 'latex' not in kwargs:
            # if not, print a warning
            print('No LaTeX labels provided: manually edit the updated.yaml file to avoid lenstronomy labels breaking GetDist.')
            pass
        else:
            latex = kwargs['latex']
            # check the right number of labels are present
            if len(latex) != len(sampled_params.keys()):
                # if not, raise ValueError
                raise ValueError('You must provide the same number of labels as sampled parameters.')
            # update sampled_params dict with labels
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
        # if not present, passes a default value (most taken from cobaya docs)
        # note: parameter blocking kwargs not provided because fast/slow parameters do not exist in strong lensing
        # todo: remove drag and oversample options?
        # also the temperature option is apparently deprecated

        mcmc_kwargs = {'burn_in': kwargs.get('burn_in', 0),
                       'max_tries': kwargs.get('max_tries', 100*self._num_params),
                       'covmat': kwargs.get('covmat', None),
                       'proposal_scale': kwargs.get('proposal_scale', 1),
                       'output_every': kwargs.get('output_every', 500),
                       'learn_every': kwargs.get('learn_every', 40*self._num_params),
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
            print('Sampling failed!')

        # get the best fit (max likelihood); returns a pandas series
        # we use the native cobaya calculation instead of lenstronomy's
        # this is because crun does not directly expose the samples themselves
        best_fit_series = sampler.collection.bestfit()

        # turn that pandas series into a list (of floats)
        best_fit_values = best_fit_series[sampled_params].values.tolist()

        return updated_info, sampler, best_fit_values
