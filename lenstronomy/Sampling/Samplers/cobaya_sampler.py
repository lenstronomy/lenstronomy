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

        # self._kwargs_temp = self._updateManager.parameter_state
        # self._mean_start = self._ll.param.kwargs2args(**self._kwargs_temp)

    def run(self, **kwargs):
        """
        docstring goes here
        """

        from cobaya.run import run as crun

        sampled_params = self._param_names

        sampled_params = {k: {'prior': {'min': self._lower_limit[i], 'max': self._upper_limit[i]}} for k, i in zip(sampled_params, range(len(sampled_params)))} # add the priors to the sampled_params

        # refs = self._mean_start

        # print(refs)

        # refs = [1.5, 0.3, 3.0, 0.1, 0.1]

        # add reference values to start chain close to expected best fit
        # [sampled_params[k].update({'ref': refs[i]}) for k, i in zip(sampled_params.keys(), range(len(refs)))]

        # add LaTeX labels so lenstronomy kwarg names don't break getdist plotting
        if kwargs['latex'] is not None:
            latex = kwargs['latex']
            [sampled_params[k].update({'latex': latex[i]}) for k, i in zip(sampled_params.keys(), range(len(latex)))]

        def likelihood_for_cobaya(**kwargs):
            current_input_values = [kwargs[p] for p in sampled_params]
            logp = self._ll.likelihood(current_input_values)
            return logp

        info_like = {"lenstronomy_likelihood": {"external": likelihood_for_cobaya, "input_params": sampled_params}}#, "output_params": ["sum_a"]}}

        info = {'likelihood': info_like}

        info['params'] = sampled_params

        # Gelman--Rubin criterion
        GR = kwargs['GR']

        # max attempts to start the chain
        mt = kwargs['max_tries']

        info['sampler'] = {'mcmc': {'Rminus1_stop': GR, 'max_tries': mt}} # consider what should be hardcoded here or not

        info['output'] = kwargs['path']

        updated_info, sampler = crun(info)

        output = [updated_info, sampler]

        return output
