from lenstronomy.Extensions.Itterative.iterative_psf import PSF_iterative
from lenstronomy.Workflow.fitting import Fitting


class FittingSequence(object):
    """
    class to define a sequence of fitting applied, inherite the Fitting class
    """
    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_init, kwargs_sigma, kwargs_fixed):
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.kwargs_options = kwargs_options
        self._lens_init, self._source_init, self._lens_light_init, self._else_init = kwargs_init
        self._lens_sigma, self._source_sigma, self._lens_light_sigma, self._else_sigma = kwargs_sigma
        self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._else_fixed = kwargs_fixed

        self.fitting = Fitting(kwargs_data=kwargs_data, kwargs_psf=kwargs_psf, kwargs_lens_fixed=self._lens_fixed,
                          kwargs_source_fixed=self._source_fixed, kwargs_lens_light_fixed=self._lens_light_fixed,
                          kwargs_else_fixed=self._else_fixed)

        self.psf_iter = PSF_iterative()

    def fit_sequence(self, fitting_kwargs_list):
        """

        :param fitting_kwargs_list: list of kwargs specify the fitting routine to be executed
        :return:
        """
        chain_list = []
        param_list = []
        samples_mcmc, param_mcmc, dist_mcmc = [], [], []
        lens_temp, source_temp, lens_light_temp, else_temp = self._init_kwargs()
        for fitting_kwargs in fitting_kwargs_list:
            if fitting_kwargs['fitting_routine'] == 'MCMC':
                samples_mcmc, param_mcmc, dist_mcmc = self.mcmc(fitting_kwargs, lens_temp, source_temp, lens_light_temp, else_temp)
            else:
                lens_temp, source_temp, lens_light_temp, else_temp, chain, param = self.fit_single(fitting_kwargs,
                    lens_temp, source_temp, lens_light_temp, else_temp)
                chain_list.append(chain)
                param_list.append(param)
            # print statement, parameters ect
        #TODO update fixed parameters as in fitting class
        return lens_temp, source_temp, lens_light_temp, else_temp, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc

    def mcmc(self, fitting_kwargs, lens_input, source_input, lens_light_input, else_input):
        """

        :param fitting_kwargs:
        :param lens_input:
        :param source_input:
        :param lens_light_input:
        :param else_input:
        :return:
        """
        n_burn = fitting_kwargs['n_burn']
        n_run = fitting_kwargs['n_run']
        walkerRatio = fitting_kwargs['walkerRatio']
        mpi = fitting_kwargs['mpi']
        sigma_scale = fitting_kwargs['sigma_scale']
        lens_sigma, source_sigma, lens_light_sigma, else_sigma = self._sigma_kwargs()
        samples, param, dist = self.fitting.mcmc_run(self.kwargs_options,
                              lens_input, source_input, lens_light_input, else_input,
                              lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                              n_burn, n_run, walkerRatio, threadCount=1, mpi=mpi, init_samples=None, sigma_factor=sigma_scale)
        return samples, param, dist

    def fit_single(self, fitting_kwargs, lens_input, source_input, lens_light_input, else_input):
        """

        :param fitting_kwargs:
        :param lens_input:
        :param source_input:
        :param lens_light_input:
        :param else_input:
        :return:
        """
        fitting_routine = fitting_kwargs['fitting_routine']
        mpi = fitting_kwargs['mpi']
        sigma_scale = fitting_kwargs['sigma_scale']
        n_particles = fitting_kwargs['n_particles']
        n_iterations = fitting_kwargs['n_iterations']
        psf_iteration = fitting_kwargs.get('psf_iteration', False)
        lens_sigma, source_sigma, lens_light_sigma, else_sigma = self._sigma_kwargs()
        if fitting_routine == 'lens_light_mask':
            lens_result, source_result, lens_light_result, else_result, chain, param_list, _ = self.fitting.find_lens_light_mask(
                self.kwargs_options, lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale)
        elif fitting_routine == 'lens_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list, _ = self.fitting.find_lens_only(
                self.kwargs_options, lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale)
        elif fitting_routine == 'source_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list, _ = self.fitting.find_source_only(
                self.kwargs_options, lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale)
        elif fitting_routine == 'lens_light_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list, _ = self.fitting.find_lens_light_only(
                self.kwargs_options, lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale)
        elif fitting_routine == 'lens_combined':
            lens_result, source_result, lens_light_result, else_result, chain, param_list, _ = self.fitting.find_lens_combined(
                self.kwargs_options, lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale)
        else:
            raise ValueError("%s is not a valid fitting routine" %fitting_routine)


        if psf_iteration is True:
            psf_iter_factor = fitting_kwargs['psf_iter_factor']
            psf_iter_num = fitting_kwargs['psf_iter_num']
            psf_symmetry = self.kwargs_options.get('psf_symmetry', 1)
            self.kwargs_psf = self.psf_iter.update_iterative(self.kwargs_data, self.kwargs_psf, self.kwargs_options, lens_result, source_result,
                                                   lens_light_result, else_result, factor=psf_iter_factor, num_iter=psf_iter_num,
                                                   symmetry=psf_symmetry, verbose=False)

        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def _init_kwargs(self):
        """

        :return: initial kwargs
        """
        return self._lens_init, self._source_init, self._lens_light_init, self._else_init

    def _sigma_kwargs(self):
        return self._lens_sigma, self._source_sigma, self._lens_light_sigma, self._else_sigma

    def _fixed_kwargs(self):
        return self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._else_fixed