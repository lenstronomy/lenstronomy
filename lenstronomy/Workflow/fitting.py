__author__ = 'sibirrer'

import numpy as np

from lenstronomy.MCMC.mcmc import MCMC_sampler
from lenstronomy.MCMC.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.parameters import Param


class Fitting(object):
    """
    class to find a good estimate of the parameter positions and uncertainties to run a (full) MCMC on
    """

    def __init__(self, kwargs_data, kwargs_psf, kwargs_lens_fixed=[], kwargs_source_fixed=[], kwargs_lens_light_fixed=[], kwargs_else_fixed={}):
        """

        :return:
        """
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.kwargs_lens_fixed = kwargs_lens_fixed # always fixed parameters
        self.kwargs_source_fixed = kwargs_source_fixed  # always fixed parameters
        self.kwargs_lens_light_fixed = kwargs_lens_light_fixed  # always fixed parameters
        self.kwargs_else_fixed = kwargs_else_fixed  # always fixed parameters

    def _run_pso(self, n_particles, n_iterations, kwargs_options, kwargs_data, kwargs_psf,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
                 threadCount=1, mpi=False, print_key='Default', sigma_factor=1):
        kwargs_prior_lens = []
        for k in range(len(kwargs_mean_lens)):
            kwargs_prior_lens.append(dict(kwargs_mean_lens[k].items() + kwargs_sigma_lens[k].items()))
        kwargs_prior_source = []
        for k in range(len(kwargs_mean_source)):
            kwargs_prior_source.append(dict(kwargs_mean_source[k].items() + kwargs_sigma_source[k].items()))
        kwargs_prior_lens_light = []
        for k in range(len(kwargs_mean_lens_light)):
            kwargs_prior_lens_light.append(dict(kwargs_mean_lens_light[k].items() + kwargs_sigma_lens_light[k].items()))
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())
        # initialise mcmc classes

        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else = self._update_fixed(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
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

    def _update_fixed(self, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else):
        param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens_updated = []
        for k in range(len(lens_fix)):
            kwargs_fixed_lens_updated.append(dict(kwargs_fixed_lens[k].items() + lens_fix[k].items()))
        kwargs_fixed_source_updated = []
        for k in range(len(source_fix)):
            kwargs_fixed_source_updated.append(dict(kwargs_fixed_source[k].items() + source_fix[k].items()))
        kwargs_fixed_lens_light_updated = []
        for k in range(len(lens_light_fix)):
            kwargs_fixed_lens_light_updated.append(dict(kwargs_fixed_lens_light[k].items() + lens_light_fix[k].items()))
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())
        return kwargs_fixed_lens_updated, kwargs_fixed_source_updated, kwargs_fixed_lens_light_updated, kwargs_fixed_else

    def _mcmc_run(self, n_burn, n_run, walkerRatio, kwargs_options, kwargs_data, kwargs_psf,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
                 threadCount=1, mpi=False, init_samples=None, sigma_factor=1):

        kwargs_prior_lens = []
        for k in range(len(kwargs_mean_lens)):
            kwargs_prior_lens.append(dict(kwargs_mean_lens[k].items() + kwargs_sigma_lens[k].items()))
        kwargs_prior_source = []
        for k in range(len(kwargs_mean_source)):
            kwargs_prior_source.append(dict(kwargs_mean_source[k].items() + kwargs_sigma_source[k].items()))
        kwargs_prior_lens_light = []
        for k in range(len(kwargs_mean_lens_light)):
            kwargs_prior_lens_light.append(dict(kwargs_mean_lens_light[k].items() + kwargs_sigma_lens_light[k].items()))
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())
        # initialise mcmc classes

        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else = self._update_fixed(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
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

    def _fixed_else(self, kwargs_options, kwargs_else):
        """

        :param kwargs_options:
        :param kwargs_else:
        :return:
        """
        num_images = kwargs_options.get('num_images', 0)
        if num_images > 0:
            kwargs_fixed = {'point_amp': np.ones(num_images)}
        else:
            kwargs_fixed = {}
        return kwargs_fixed

    def _fixed_light(self, kwargs_options, kwargs_light, type):
        """

        :param kwargs_options:
        :param kwargs_light:
        :param type:
        :return:
        """
        model_list = kwargs_options[type]
        kwargs_fixed_list = []
        if type == 'source_light_model_list':
            kwargs_fixed_global = self.kwargs_source_fixed
        else:
            kwargs_fixed_global = self.kwargs_lens_light_fixed
        for i, model in enumerate(model_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
                kwargs_fixed = {'I0_sersic': 1}
            elif model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_fixed = {'I0_sersic': 1, 'I0_2': 1}
            elif model in ['BULDGE_DISK']:
                kwargs_fixed = {'I0_b': 1, 'I0_d': 1}
            elif model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_fixed = {'sigma0': 1.}
            elif model in ['GAUSSIAN']:
                kwargs_fixed = {'amp': 1}
            elif model in ['MULTI_GAUSSIAN']:
                num = len(kwargs_fixed_global[i]['sigma'])
                kwargs_fixed = {'amp': np.ones(num)}
            elif model in ['SHAPELETS']:
                if 'n_max' in kwargs_fixed_global[i]:
                    n_max = kwargs_fixed_global[i]['n_max']
                else:
                    n_max = kwargs_light[i]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_fixed = {'amp': np.ones(num_param)}
            elif model in ['UNIFORM']:
                kwargs_fixed = {'mean': 1}
            else:
                kwargs_fixed = {}
            if type == 'source_light_model_list':
                if kwargs_options.get('solver', False) or kwargs_options.get('image_plane_source', False):
                    if kwargs_options.get('joint_center', False) or i == 0:
                        kwargs_fixed['center_x'] = 0
                        kwargs_fixed['center_y'] = 0
            kwargs_fixed_list.append(kwargs_fixed)
        return kwargs_fixed_list

    def _fixed_lens(self, kwargs_options, kwargs_lens_list):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        kwargs_fixed_lens_list = []
        for k in range(len(kwargs_lens_list)):
            if k == 0:
                if kwargs_options.get('solver', False) is True:
                    lens_model = kwargs_options['lens_model_list'][0]
                    kwargs_lens = kwargs_lens_list[0]
                    if kwargs_options['num_images'] == 4:
                        if lens_model in ['SPEP', 'SPEMD']:
                            kwargs_fixed_lens = {'theta_E': kwargs_lens['theta_E'], 'q': kwargs_lens['q'],
                                             'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                             'center_y': kwargs_lens['center_y']}
                        elif lens_model in ['NFW_ELLIPSE']:
                            kwargs_fixed_lens = {'theta_Rs': kwargs_lens['theta_Rs'], 'q': kwargs_lens['q'],
                                             'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                             'center_y': kwargs_lens['center_y']}
                        elif lens_model in ['SHAPELETS_CART']:
                            kwargs_fixed_lens = {}
                        elif lens_model in ['NONE']:
                            kwargs_fixed_lens = {}
                        else:
                            raise ValueError("%s is not a valid option. Choose from 'PROFILE', 'COMPOSITE', 'NFW_PROFILE', 'SHAPELETS'" % kwargs_options['solver_type'])
                    elif kwargs_options['num_images'] == 2:
                        if lens_model in ['SPEP', 'SPEMD', 'NFW_ELLIPSE', 'COMPOSITE']:
                            if kwargs_options['solver_type'] in ['CENTER']:
                                kwargs_fixed_lens = {'center_x': kwargs_lens['center_x'],
                                                     'center_y': kwargs_lens['center_y']}
                            elif kwargs_options['solver_type'] in ['ELLIPSE']:
                                kwargs_fixed_lens = {'phi_G': kwargs_lens['phi_G'], 'q': kwargs_lens['q']}
                            else:
                                raise ValueError("solver_type %s not valid for lens model %s" % (kwargs_options['solver_type'], lens_model))
                        elif lens_model == "SHAPELETS_CART":
                            kwargs_fixed_lens = {}
                        elif lens_model == 'EXTERNAL_SHEAR':
                            kwargs_fixed_lens = {'e1': kwargs_lens['e1'], 'e2': kwargs_lens['e2']}
                        else:
                            raise ValueError("%s is not a valid option for solver_type in combination with lens model %s" % (kwargs_options['solver_type'], lens_model))
                    else:
                        raise ValueError("%s is not a valid number of points" % kwargs_options['num_images'])
                else:
                    kwargs_fixed_lens = {}
            else:
                kwargs_fixed_lens = {}
            kwargs_fixed_lens_list.append(kwargs_fixed_lens)
        return kwargs_fixed_lens_list

    def find_lens_catalogue(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, sigma_factor=1):
        """
        finds the positon of a SPEP configuration based on the catalogue level input
        :return: constraints of lens model
        """
        kwargs_options_special = {'lens_model_list': ['ELLIPSE'], 'lens_light_model_list': ['NONE'], 'source_light_model_list': ['NONE'],
                                  'X2_type': 'catalogue', 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens.copy()
        if 'gamma' in kwargs_lens:
            kwargs_fixed_lens[0] = {'gamma': kwargs_lens['gamma']}  # for SPEP lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())

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
        if kwargs_options['lens_light_model_list'] is []:
            return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, [None]*4, None, kwargs_options
        kwargs_options_special = {'lens_model_list': ['NONE'], 'source_light_model_list': ['NONE'],
                                  'X2_type': 'image', 'solver': False, 'lens_light_mask': True, 'point_source': False, 'additional_images': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = kwargs_source
        #source_fixed = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        #for k in range(len(kwargs_source)):
        #    kwargs_fixed_source.append(dict(kwargs_source[k].items() + source_fixed[k].items()))
        kwargs_fixed_lens_light = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens light mask', sigma_factor=sigma_factor)
        return kwargs_lens, kwargs_source, lens_light_result, kwargs_else, chain, param_list, kwargs_options_execute

    def find_lens_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens model with fixed lens light model, type as specified in input kwargs_optinons
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image'}

        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())

        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        if 'gamma' in kwargs_lens[0]:
            kwargs_fixed_lens[0]['gamma'] = kwargs_lens[0]['gamma']
        kwargs_fixed_source = []
        source_fixed = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        for k in range(len(kwargs_source)):
            kwargs_fixed_source.append(dict(kwargs_source[k].items() + source_fixed[k].items()))
        kwargs_fixed_lens_light = []
        lens_light_fixed = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        for k in range(len(kwargs_lens_light)):
            kwargs_fixed_lens_light.append(dict(kwargs_lens_light[k].items() + lens_light_fixed[k].items()))
        kwargs_fixed_else = self._fixed_else(kwargs_options, kwargs_else)

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
        kwargs_options_special = {'X2_type': 'image'}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = []
        source_fixed = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        for k in range(len(kwargs_source)):
            kwargs_fixed_source.append(dict(kwargs_source[k].items() + source_fixed[k].items()))
        kwargs_fixed_lens_light = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())

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
        kwargs_options_special = {'X2_type': 'image'}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_fixed_lens_light = []
        lens_light_fixed = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        for k in range(len(kwargs_lens_light)):
            kwargs_fixed_lens_light.append(dict(kwargs_lens_light[k].items() + lens_light_fixed[k].items()))
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='source light', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_fixed_lens(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        finds lens light and source light combined with fixed lens model
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image'}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_fixed_lens_light = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        kwargs_fixed_else = self._fixed_else(kwargs_options, kwargs_else)

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='lens fixed', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_lens_combined(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1, gamma_fixed=False):
        """
        finds lens light and lens model combined fit
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image'}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        if gamma_fixed:
            if 'gamma' in kwargs_lens[0]:
                kwargs_fixed_lens[0]['gamma'] = kwargs_lens[0]['gamma']
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_fixed_lens_light = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        kwargs_fixed_else = self._fixed_else(kwargs_options, kwargs_else)

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='combined', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_buldge_disk(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1):
        """
        fits a buldge-to-disk decomposition to the source galaxy
        :return: constraints of lens model
        """
        kwargs_options_special = {'X2_type': 'image', 'source_light_model_list': ['BULDGE_DISK']}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_source_new = [{'center_x': kwargs_source[0]['center_x'], 'center_y': kwargs_source[0]['center_y']
                                 , 'R_b': 0.05, 'phi_G_b':0, 'q_b':1., 'R_d': 0.1, 'phi_G_d':0, 'q_d':1.}]
        kwargs_source_sigma_new = [{'center_x_sigma': 0.001, 'center_y_sigma': 0.001, 'R_b_sigma': 0.01
                                       , 'ellipse_sigma': 0.1, 'R_d_sigma': 0.05}]
        kwargs_fixed_lens_light = []
        lens_light_fixed = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        for k in range(len(kwargs_lens_light)):
            kwargs_fixed_lens_light.append(dict(kwargs_lens_light[k].items() + lens_light_fixed[k].items()))
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source_new, kwargs_source_sigma_new,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, print_key='buldge-disk', sigma_factor=sigma_factor)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def mcmc_buldge_disk(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                 kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=False, init_samples=None, sigma_factor=1):
        """
        MCMC
        """
        kwargs_options_special = {'X2_type': 'image', 'source_light_model_list': ['BULDGE_DISK'], 'solver': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_source_sigma_new = [{'center_x_sigma': 0.001, 'center_y_sigma': 0.001, 'R_b_sigma': 0.05
                                       , 'ellipse_sigma': 0.1, 'R_d_sigma': 0.05}]
        kwargs_fixed_lens_light = []
        lens_light_fixed = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        for k in range(len(kwargs_lens_light)):
            kwargs_fixed_lens_light.append(dict(kwargs_lens_light[k].items() + lens_light_fixed[k].items()))
        kwargs_fixed_else = dict(kwargs_else.items() + self._fixed_else(kwargs_options, kwargs_else).items())
        samples, param_list, dist = self._mcmc_run(
            n_burn, n_run, walkerRatio, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma_new,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, init_samples=init_samples, sigma_factor=sigma_factor)
        return samples, param_list, dist


    def mcmc_run(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                 kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=False, init_samples=None, sigma_factor=1):
        """
        MCMC
        """
        kwargs_options_execute, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else = self._mcmc_run_fixed(kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        samples, param_list, dist = self._mcmc_run(
            n_burn, n_run, walkerRatio, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi=mpi, init_samples=init_samples, sigma_factor=sigma_factor)
        return samples, param_list, dist

    def _mcmc_run_fixed(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
        """

        :param kwargs_options:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        kwargs_options_special = {'X2_type': 'image'}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = self._fixed_light(kwargs_options_execute, kwargs_source, 'source_light_model_list')
        kwargs_fixed_lens_light = []
        lens_light_fixed = self._fixed_light(kwargs_options_execute, kwargs_lens_light, 'lens_light_model_list')
        for k in range(len(kwargs_lens_light)):
            kwargs_fixed_lens_light.append(dict(kwargs_lens_light[k].items() + lens_light_fixed[k].items()))
        kwargs_fixed_else = self._fixed_else(kwargs_options, kwargs_else)
        return kwargs_options_execute, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else