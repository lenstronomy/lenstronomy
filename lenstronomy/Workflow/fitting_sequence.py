from lenstronomy.ImSim.psf_fitting import PSF_fitting
from lenstronomy.Workflow.fitting import Fitting
from lenstronomy.MCMC.alignment_matching import AlignmentFitting
import lenstronomy.Util.class_creator as class_creator

class FittingSequence(object):
    """
    class to define a sequence of fitting applied, inherite the Fitting class
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params):
        self.multi_band_list = multi_band_list
        self.kwargs_model = kwargs_model
        self.kwargs_constraints = kwargs_constraints
        self.kwargs_likelihood = kwargs_likelihood
        kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper = kwargs_params
        self._lens_init, self._source_init, self._lens_light_init, self._else_init = kwargs_init
        self._lens_sigma, self._source_sigma, self._lens_light_sigma, self._else_sigma = kwargs_sigma
        self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._else_fixed = kwargs_fixed

        self.fitting = Fitting(multi_band_list=multi_band_list, kwargs_model=kwargs_model,
                               kwargs_constraints=kwargs_constraints, kwargs_likelihood=kwargs_likelihood,
                               kwargs_fixed=kwargs_fixed,
                               kwargs_lower=kwargs_lower, kwargs_upper=kwargs_upper)

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
            if fitting_kwargs['fitting_routine'] in ['MCMC', 'MCMC_source']:
                samples_mcmc, param_mcmc, dist_mcmc = self.mcmc(fitting_kwargs, lens_temp, source_temp, lens_light_temp, else_temp)
            else:
                lens_temp, source_temp, lens_light_temp, else_temp, chain, param = self.fit_single(fitting_kwargs,
                    lens_temp, source_temp, lens_light_temp, else_temp)
                chain_list.append(chain)
                param_list.append(param)
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
        mpi = fitting_kwargs.get('mpi', False)
        sigma_scale = fitting_kwargs['sigma_scale']
        lens_sigma, source_sigma, lens_light_sigma, else_sigma = self._sigma_kwargs()
        compute_bool = fitting_kwargs.get('compute_bands', None)
        if fitting_kwargs['fitting_routine'] == 'MCMC':
            samples, param, dist = self.fitting.mcmc_run(
                                  lens_input, source_input, lens_light_input, else_input,
                                  lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                                  n_burn, n_run, walkerRatio, threadCount=1, mpi=mpi, init_samples=None,
                                  sigma_factor=sigma_scale, gamma_fixed=False, compute_bool=compute_bool, fix_lens=False,
                                  fix_source=False, fix_lens_light=False, fix_point_source=False)
        elif fitting_kwargs['fitting_routine'] == 'MCMC_source':
            samples, param, dist = self.fitting.mcmc_run(
                                  lens_input, source_input, lens_light_input, else_input,
                                  lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                                  n_burn, n_run, walkerRatio, threadCount=1, mpi=mpi, init_samples=None,
                                  sigma_factor=sigma_scale, gamma_fixed=False, compute_bool=compute_bool, fix_lens=True,
                                  fix_source=False, fix_lens_light=True, fix_point_source=True)
        else:
            raise ValueError("%s is not supported as a mcmc routine" % fitting_kwargs['fitting_routine'])
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
        mpi = fitting_kwargs.get('mpi', False)
        sigma_scale = fitting_kwargs.get('sigma_scale', 1)
        n_particles = fitting_kwargs.get('n_particles', 10)
        n_iterations = fitting_kwargs.get('n_iterations', 10)
        compute_bool = fitting_kwargs.get('compute_bands', [True]*len(self.multi_band_list))
        lens_sigma, source_sigma, lens_light_sigma, else_sigma = self._sigma_kwargs()

        if fitting_routine == 'lens_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=False,
                fix_lens=False, fix_source=True, fix_lens_light=True, fix_point_source=False, print_key=fitting_routine)
        elif fitting_routine == 'source_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=False,
                fix_lens=True, fix_source=False, fix_lens_light=True, fix_point_source=True, print_key=fitting_routine)
        elif fitting_routine == 'lens_light_only':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=False,
                fix_lens=True, fix_source=True, fix_lens_light=False, fix_point_source=True, print_key=fitting_routine)
        elif fitting_routine == 'lens_fixed':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=False,
                fix_lens=True, fix_source=False, fix_lens_light=False, fix_point_source=False, print_key=fitting_routine)
        elif fitting_routine == 'lens_combined_gamma_fixed':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=True,
                fix_lens=False, fix_source=False, fix_lens_light=False, fix_point_source=False, print_key=fitting_routine)
        elif fitting_routine == 'lens_combined':
            lens_result, source_result, lens_light_result, else_result, chain, param_list = self.fitting.pso_run(
                lens_input, source_input, lens_light_input, else_input,
                lens_sigma, source_sigma, lens_light_sigma, else_sigma,
                n_particles, n_iterations, mpi=mpi, sigma_factor=sigma_scale, compute_bool=compute_bool, gamma_fixed=False,
                fix_lens=False, fix_source=False, fix_lens_light=False, fix_point_source=False, print_key=fitting_routine)
        elif fitting_routine == 'psf_iteration':
            print('PSF fitting...')
            psf_iter_factor = fitting_kwargs['psf_iter_factor']
            psf_iter_num = fitting_kwargs['psf_iter_num']
            for i in range(len(self.multi_band_list)):
                if compute_bool[i]:
                    kwargs_data = self.multi_band_list[i][0]
                    kwargs_psf = self.multi_band_list[i][1]
                    kwargs_numerics = self.multi_band_list[i][2]
                    psf_symmetry = kwargs_psf.get('psf_symmetry', 1)
                    image_model = class_creator.creat_image_model(kwargs_data=kwargs_data,
                                                                  kwargs_psf=kwargs_psf,
                                                                  kwargs_numerics=kwargs_numerics,
                                                                  kwargs_model=self.kwargs_model)
                    psf_iter = PSF_fitting(image_model_class=image_model)
                    kwargs_psf = psf_iter.update_iterative(kwargs_psf, lens_input, source_input,
                                                                    lens_light_input, else_input,
                                                                    factor=psf_iter_factor, num_iter=psf_iter_num,
                                                                    symmetry=psf_symmetry, verbose=True)
                    self.multi_band_list[i][1] = kwargs_psf
            lens_result, source_result, lens_light_result, else_result = lens_input, source_input, lens_light_input, else_input
            chain, param_list = [], []
            print('PSF fitting completed')
        elif fitting_routine == 'align_images':
            print('Align images...')
            lens_result, source_result, lens_light_result, else_result = lens_input, source_input, lens_light_input, else_input
            for i in range(len(self.multi_band_list)):
                if compute_bool[i]:
                    kwargs_data = self.multi_band_list[i][0]
                    kwargs_psf = self.multi_band_list[i][1]
                    kwargs_numerics = self.multi_band_list[i][2]
                    alignmentFitting = AlignmentFitting(kwargs_data, kwargs_psf, kwargs_numerics, self.kwargs_model, lens_input, source_input,
                                                                    lens_light_input, else_input, compute_bool=compute_bool)
                    lowerLimit = fitting_kwargs.get('lower_limit_shift', -0.2)
                    upperLimit = fitting_kwargs.get('upper_limit_shift', 0.2)
                    kwargs_data, chain = alignmentFitting.pso(n_particles, n_iterations, lowerLimit, upperLimit,
                                                              threadCount=1, mpi=mpi,
                                                              print_key='Alignment fitting for band %s ...' % i)
                    print('Align completed for band %s.' % i)
                    self.multi_band_list[i][0] = kwargs_data
            chain, param_list = [], []
        else:
            raise ValueError("%s is not a valid fitting routine" %fitting_routine)

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