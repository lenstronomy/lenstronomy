__author__ = 'sibirrer'

import numpy as np
from lenstronomy.MCMC.mcmc import MCMC_sampler
from lenstronomy.MCMC.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.parameters import Param


class Fitting(object):
    """
    class to find a good estimate of the parameter positions and uncertainties to run a (full) MCMC on
    """

    def __init__(self, kwargs_data, kwargs_psf, kwargs_lens_fixed={}, kwargs_source_fixed={}, kwargs_lens_light_fix={}, kwargs_else_fixed={}):
        """

        :return:
        """
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.kwargs_lens_fixed = kwargs_lens_fixed # always fixed parameters
        self.kwargs_source_fixed = kwargs_source_fixed  # always fixed parameters
        self.kwargs_lens_light_fixed = kwargs_lens_light_fix  # always fixed parameters
        self.kwargs_else_fixed = kwargs_else_fixed  # always fixed parameters

    def _run_pso(self, n_particles, n_iterations, kwargs_options, kwargs_data, kwargs_psf,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
                 threadCount=1, mpi=False, print_key='Default', sigma_factor=1):
        kwargs_prior_lens = dict(kwargs_mean_lens.items() + kwargs_sigma_lens.items())
        kwargs_prior_source = dict(kwargs_mean_source.items() + kwargs_sigma_source.items())
        kwargs_prior_lens_light = dict(kwargs_mean_lens_light.items() + kwargs_sigma_lens_light.items())
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())
        # initialise mcmc classes

        param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_source = dict(kwargs_fixed_source.items() + source_fix.items())
        kwargs_fixed_lens_light = dict(kwargs_fixed_lens_light.items() + lens_light_fix.items())
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())
        param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_else)
        lowerLimit = np.array(mean_start) - np.array(sigma_start)*sigma_factor
        upperLimit = np.array(mean_start) + np.array(sigma_start)*sigma_factor
        num_param, param_list = param_class.num_param()
        init_pos = param_class.setParams(kwargs_mean_lens, kwargs_mean_source,
                                         kwargs_mean_lens_light, kwargs_mean_else)
        # run PSO
        mcmc_class = MCMC_sampler(kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                                kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_result, source_result, lens_light_result, else_result, chain = mcmc_class.pso(n_particles,
                                                                                                       n_iterations,
                                                                                                       lowerLimit,
                                                                                                       upperLimit,
                                                                                                       init_pos=init_pos,
                                                                                                       threadCount=threadCount,
                                                                                                       mpi=mpi,
                                                                                                       print_key=print_key)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def _mcmc_run(self, n_burn, n_run, walkerRatio, kwargs_options, kwargs_data, kwargs_psf,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
                 threadCount=1, mpi=False, init_samples=None, sigma_factor=1):


        kwargs_prior_lens = dict(kwargs_mean_lens.items() + kwargs_sigma_lens.items())
        kwargs_prior_source = dict(kwargs_mean_source.items() + kwargs_sigma_source.items())
        kwargs_prior_lens_light = dict(kwargs_mean_lens_light.items() + kwargs_sigma_lens_light.items())
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())
        # initialise mcmc classes

        param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_source = dict(kwargs_fixed_source.items() + source_fix.items())
        kwargs_fixed_lens_light = dict(kwargs_fixed_lens_light.items() + lens_light_fix.items())
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())

        mcmc_class = MCMC_sampler(kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                                kwargs_fixed_lens_light, kwargs_fixed_else)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_else)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_samples is None:
            initpos = ReusePositionGenerator(init_samples)
        else:
            initpos = None

        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, np.array(sigma_start)*sigma_factor, threadCount=threadCount,
                                           mpi=mpi, init_pos=initpos)
        return samples, param_list, dist

    def _fixed_lens_light(self, kwargs_options):
        """

        :param kwargs_options:
        :return: fixed linear parameters in lens light function
        """
        if kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif kwargs_options['lens_light_type'] in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        elif kwargs_options['lens_light_type'] in ['DOUBLE_SERSIC', 'DOULBE_CORE_SERSIC']:
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1}
        else:
            kwargs_fixed_lens_light = {}
        return kwargs_fixed_lens_light

    def _fixed_source(self, kwargs_options):
        """

        :param kwargs_options:
        :return: fixed linear parameters in lens light function
        """
        if kwargs_options['source_type'] == 'TRIPPLE_SERSIC':
            kwargs_fixed_source = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif kwargs_options['source_type'] in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
            kwargs_fixed_source = {'I0_sersic': 1}
        elif kwargs_options['source_type'] in ['DOUBLE_SERSIC', 'DOULBE_CORE_SERSIC']:
            kwargs_fixed_source = {'I0_sersic': 1, 'I0_2': 1}
        else:
            kwargs_fixed_source = {}
        return kwargs_fixed_source

    def _fixed_lens(self, kwargs_options, kwargs_lens):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        if kwargs_options['solver'] is True:
            if kwargs_options['solver_type'] in ['SPEP', 'SPEMD']:
                if kwargs_options['num_images'] == 4:
                    kwargs_fixed_lens = {'theta_E': kwargs_lens['theta_E'], 'q': kwargs_lens['q'],
                                     'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                     'center_y': kwargs_lens['center_y']}
                elif kwargs_options['num_images'] == 2:
                    kwargs_fixed_lens = {'center_x': kwargs_lens['center_x'], 'center_y': kwargs_lens['center_y']}
                else:
                    raise ValueError("%s is not a valid option" % kwargs_options['num_images'])
            elif kwargs_options['solver_type'] == "SHAPELETS":
                kwargs_fixed_lens = {}
            else:
                raise ValueError("%s is not a valid option" % kwargs_options['solver_type'])
        else:
            kwargs_fixed_lens = {}
        return kwargs_fixed_lens

    def find_lens_catalogue(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, sigma_factor=1):
        """
        finds the positon of a SPEP configuration based on the catalogue level input
        :return: constraints of lens model
        """
        kwargs_options_special = {'lens_type': 'ELLIPSE', 'lens_light_type': 'NONE', 'source_type': 'NONE',
                                  'X2_type': 'catalogue', 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = {'gamma': kwargs_lens['gamma']}  # for SPEP lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=1, mpi=mpi, print_key='Catalogue', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def find_lens_light_mask(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens light, type as specified in input kwargs_optinons
        :return: constraints of lens model
        """
        if kwargs_options['lens_light_type'] is 'NONE':
            return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, [0,0,0,0], 0, kwargs_options
        kwargs_options_special = {'lens_type': 'NONE', 'source_type': 'NONE',
                                  'X2_type': 'lens_light', 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = dict(kwargs_source.items() + self._fixed_source(kwargs_options_execute).items())
        kwargs_fixed_lens_light = self._fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens light', sigma_factor=sigma_factor)
        return kwargs_lens, kwargs_source, lens_light_result, kwargs_else, chain, param_list, kwargs_options_execute

    def find_lens_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens model with fixed lens light model, type as specified in input kwargs_optinons
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type']}

        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())

        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = dict(kwargs_source.items() + self._fixed_source(kwargs_options_execute).items())
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens only', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_lens_light_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens light with fixed lens model
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image', 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = dict(kwargs_source.items() + self._fixed_source(kwargs_options_execute).items())
        kwargs_fixed_lens_light = self._fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens light', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_source_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens light with fixed lens model
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image', 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = self._fixed_source(kwargs_options_execute)
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens light', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_lens_combined(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens light and lens model combined fit
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type']}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = dict(kwargs_source.items() + self._fixed_source(kwargs_options_execute).items())
        kwargs_fixed_lens_light = self._fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='combined', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def mcmc_run(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                 kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=False, init_samples=None, sigma_factor=1):
        """
        MCMC
        """
        kwargs_options_special = {'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type']}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = dict(kwargs_source.items() + self._fixed_source(kwargs_options_execute).items())
        kwargs_fixed_lens_light = self._fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        samples, param_list, dist = self._mcmc_run(
            n_burn, n_run, walkerRatio, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, init_samples=init_samples, sigma_factor=sigma_factor)
        return samples, param_list, dist