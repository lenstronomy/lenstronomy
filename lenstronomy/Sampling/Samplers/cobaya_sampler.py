__author__ = 'nataliehogg'

# man with one sampling method always knows his posterior distribution; man with two never certain.

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

        from cobaya.run import run as crun

        # get the parameters to be sampled
        sampled_params = self._param_names

        # add the priors to the sampled_params
        sampled_params = {k: {'prior': {'min': self._lower_limit[i], 'max': self._upper_limit[i]}} for k, i in zip(sampled_params, range(len(sampled_params)))}

        # add reference values to start chain close to expected best fit
        # we could get these values internally from lenstronomy e.g. from _ll.param
        # that would mimic how the emcee implementation is done (with sampling_util/sample_ball_truncated())
        # but I like having it directly accessible/controllable by the user when they run the MCMC
        if 'starting_points' not in kwargs:
            print('No starting point provided. Drawing a starting point from the prior.')
            pass
        else:
            refs = kwargs['starting_points']
            if len(refs) != len(sampled_params.keys()):
                raise ValueError('You must provide the same number of starting points as sampled parameters.')
            [sampled_params[k].update({'ref': refs[i]}) for k, i in zip(sampled_params.keys(), range(len(refs)))]

        # add proposal widths
        if 'proposal_widths' not in kwargs:
            print('No proposal widths provided. Learning covariance matrix.')
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

        # select mcmc as the sampler and pass the relevant sampler settings
        info['sampler'] = {'mcmc': {'Rminus1_stop': kwargs['GR'], 'max_tries': kwargs['max_tries']}}

        # where the chains and other files will be saved
        info['output'] = kwargs['path']

        # whether or not to overwrite previous chains with the same name (bool)
        info['force'] = kwargs['force_overwrite']

        # run the sampler
        updated_info, sampler = crun(info)

        # get the best fit (max likelihood); format returned is a pandas series
        # this bypasses lenstronomy's way of doing it but matches lenstronomy result
        best_fit_series = sampler.collection.bestfit()

        # turn that pandas series into a list (of floats)
        best_fit_values = best_fit_series[sampled_params].values.tolist()

        return updated_info, sampler, best_fit_values
