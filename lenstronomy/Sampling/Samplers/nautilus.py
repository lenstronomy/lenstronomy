# Nautilus sampling
# https://github.com/johannesulf/nautilus

from lenstronomy.Sampling.Pool.pool import choose_pool
import time

__all__ = ["Nautilus"]


class Nautilus(object):
    def __init__(self, likelihood_module):
        """
        sampling with Nautilus [1]_

        References:
        -----------
        .. [1] Johannes Lange, in prep, https://github.com/johannesulf/nautilus

        :param likelihood_module: LikelihoodModule() instance
        """
        self._likelihood_module = likelihood_module
        self._num_param, _ = self._likelihood_module.param.num_param()
        (
            self._lower_limit,
            self._upper_limit,
        ) = self._likelihood_module.param.param_limits()

    def nautilus_sampling(
        self,
        prior_type="uniform",
        mpi=False,
        thread_count=1,
        verbose=True,
        one_step=False,
        **kwargs_nautilus
    ):
        """

        :param prior_type: string; prior type. Currently only 'uniform' supported
         (in addition to Prior class in Likelihood module)
        :param mpi: MPI option (currently not supported)
        :param thread_count: integer; multi-threading option (currently not supported)
        :param verbose: verbose statements of Nautilus
        :param one_step: boolean, if True, only runs one iteration of filling the sampler and re-training.
         This is meant for test purposes of the sampler to operate with little computational effort
        :param kwargs_nautilus: additional keyword arguments for Nautilus
        :return: points, log_w, log_l, log_z
        """
        from nautilus import Prior, Sampler

        prior = Prior()
        # TODO better prior integration with Nautilus
        if prior_type == "uniform":
            for i in range(self._num_param):
                prior.add_parameter(dist=(self._lower_limit[i], self._upper_limit[i]))
        else:
            raise ValueError(
                "prior_type %s is not supported for Nautilus wrapper." % prior_type
            )
        # loop through prior
        pool = choose_pool(mpi=mpi, processes=thread_count, use_dill=True)
        sampler = Sampler(
            prior,
            likelihood=self.likelihood,
            pool=pool,
            pass_dict=False,
            **kwargs_nautilus
        )
        time_start = time.time()
        if one_step is True:
            sampler.add_bound()
            sampler.fill_bound()
        else:
            sampler.run(verbose=verbose)
        points, log_w, log_l = sampler.posterior(return_as_dict=False)
        log_z = sampler.evidence()
        time_end = time.time()
        if pool.is_master():
            print(time_end - time_start, "time taken for MCMC sampling")
        return points, log_w, log_l, log_z

    def likelihood(self, args):
        """
        log likelihood

        :param args: ctype
        :return: log likelihood
        """
        python_list = []
        for i in range(self._num_param):
            python_list.append(args[i])
        return self._likelihood_module.likelihood(a=python_list)
