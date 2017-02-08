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
                 threadCount=1, mpi_monch=False, print_key='Default'):
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
        lowerLimit = np.array(mean_start) - np.array(sigma_start)
        upperLimit = np.array(mean_start) + np.array(sigma_start)
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
                                                                                                       mpi_monch=mpi_monch,
                                                                                                       print_key=print_key)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def _set_fixed_lens_light(self, kwargs_options):
        """

        :param kwargs_options:
        :return: fixed linear parameters in lens light function
        """
        if kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif kwargs_options['lens_light_type'] == 'SERSIC' or kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        return kwargs_fixed_lens_light

    def _fixed_lens(self, kwargs_options, kwargs_lens):
        """
        returns kwargs that are kept fixed during run, depending on options
        :param kwargs_options:
        :param kwargs_lens:
        :return:
        """
        if kwargs_options['solver'] is True:
            if kwargs_options['solver_type'] == "SPEP" or kwargs_options['solver_type'] == "SPEMD":
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
                             n_particles, n_iterations, mpi_monch=False):
        """
        finds the positon of a SPEP configuration based on the catalogue level input
        :return: constraints of lens model
        """
        kwargs_options_special = {'lens_type': 'ELLIPSE', 'lens_light_type': 'NONE', 'source_type': 'NONE',
                                  'X2_type': 'catalogue', 'solver': False, 'fix_source': True}
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
            threadCount=1, mpi_monch=mpi_monch, print_key='Catalogue')
        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def find_lens_light_mask(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi_monch=False, threadCount=1):
        """
        finds lens light, type as specified in input kwargs_optinons
        :return: constraints of lens model
        """
        kwargs_options_special = {'lens_type': 'NONE', 'source_type': 'NONE',
                                  'X2_type': 'lens_light', 'solver': False, 'fix_source': True}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = {}
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = self._set_fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi_monch=mpi_monch, print_key='lens light')
        return kwargs_lens, kwargs_source, lens_light_result, kwargs_else, chain, param_list, kwargs_options_execute

    def find_lens_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi_monch=False, threadCount=1):
        """
        finds lens model with fixed lens light model, type as specified in input kwargs_optinons
        :return: constraints of lens model
        """
        kwargs_options_special = {'source_type': 'NONE',
                                  'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type'],
                                  'fix_source': False, 'shatelets_off': False}

        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())

        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi_monch=mpi_monch, print_key='lens only')
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_lens_light_only(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi_monch=False, threadCount=1):
        """
        finds lens light with fixed lens model
        :return: constraints of lens model
        """
        kwargs_options_special = {'source_type': 'NONE',
                                  'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type'],
                                  'fix_source': False, 'shatelets_off': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = self._set_fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = kwargs_else

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi_monch=mpi_monch, print_key='lens light')
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_lens_light_combined(self, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                             kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                             n_particles, n_iterations, mpi_monch=False, threadCount=1):
        """
        finds lens light and lens model combined fit
        :return: constraints of lens model
        """
        kwargs_options_special = {'source_type': 'NONE',
                                  'X2_type': 'image', 'solver': True, 'solver_type': kwargs_options['ellipse_type'],
                                  'fix_source': False, 'shatelets_off': False}
        # this are the parameters which are held constant while sampling
        kwargs_options_execute = dict(kwargs_options.items() + kwargs_options_special.items())
        kwargs_fixed_lens = self._fixed_lens(kwargs_options_execute, kwargs_lens)
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_lens_light = self._set_fixed_lens_light(kwargs_options_execute)
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        lens_result, source_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations, kwargs_options_execute, self.kwargs_data, self.kwargs_psf,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_else, kwargs_else, kwargs_else_sigma,
            threadCount=threadCount, mpi_monch=mpi_monch, print_key='lens light')
        return lens_result, source_result, lens_light_result, else_result, chain, param_list, kwargs_options_execute

    def find_param_psf_iteration(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                 n_iterations, num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        # only execute after having modeled the lens with the right options
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        kwargs_fixed_lens = kwargs_lens.copy()

        if self.kwargs_options.get('multiBand', False):
            kwargs_data = self.kwargs_data[0]
            kwargs_psf = self.kwargs_psf_init["image1"]
        else:
            kwargs_data = self.kwargs_data
            kwargs_psf = self.kwargs_psf_init
        from lenstronomy.ImSim.make_image import MakeImage
        from astrofunc.util import Util_class
        util_class = Util_class()
        makeImage = MakeImage(self.kwargs_options, kwargs_data)
        x_grid, y_grid = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], subgrid_res)
        numPix = len(kwargs_data['image_data'])
        grid_final, error_map, cov_param, param = makeImage.make_image_ideal(x_grid, y_grid, kwargs_lens, kwargs_source,
                                                                             kwargs_psf, kwargs_lens_light, kwargs_else,
                                                                             numPix, kwargs_data['deltaPix'],
                                                                             subgrid_res)
        amp = makeImage.get_image_amplitudes(param, kwargs_else)
        amp /= amp[0]
        self.kwargs_options['psf_iteration'] = True
        # this are the parameters which are held constant while sampling

        kwargs_fixed_source = kwargs_source.copy()
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light.copy()
        kwargs_fixed_else = kwargs_else.copy()

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = {}
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = kwargs_else
        kwargs_mean_else['point_amp'] = amp
        # sigma values
        kwargs_sigma_lens = {}
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='psf iteration')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_all_arc(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                           num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        """

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param n_particles:
        :param n_iterations:
        :param num_order:
        :param subgrid_res:
        :param numThreads:
        :param mpi_monch:
        :return:
        """
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        # this are the parameters which are held constant while sampling

        kwargs_fixed_lens = {}
        # del kwargs_fixed_lens['gamma']
        kwargs_fixed_source = {'center_x': kwargs_source['center_x'], 'center_y': kwargs_source['center_y'],
                               'I0_sersic': 1}
        kwargs_fixed_psf = self.kwargs_psf_init
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}
        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = kwargs_lens_light
        if not "gamma1_foreground" in kwargs_else or not "gamma2_foreground" in kwargs_else:
            kwargs_else = dict(self.kwargs_else_init.items() + kwargs_else.items())
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = self.kwargs_lens_sigma_constraint
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='all arc')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_all_arc_point(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                 n_iterations, num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        """

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param n_particles:
        :param n_iterations:
        :param num_order:
        :param subgrid_res:
        :param numThreads:
        :param mpi_monch:
        :return:
        """
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        # this are the parameters which are held constant while sampling

        kwargs_fixed_lens = {}
        # del kwargs_fixed_lens['gamma']
        kwargs_fixed_source = {'center_x': kwargs_source['center_x'], 'center_y': kwargs_source['center_y'],
                               'I0_sersic': 1}
        kwargs_fixed_psf = self.kwargs_psf_init
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = self.kwargs_lens_sigma_constraint
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='all arc')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_perturb(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                           num_order, subgrid_res=2, numThreads=1, mpi_monch=False, clump='NFW'):
        """
        finds a lens configuration such that the catalogue is matched with a given subclump configuration
        """
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['solver_type'] = self.kwargs_options['ellipse_type']
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        if clump == 'NFW':
            self.kwargs_options['lens_type'] = 'SPEP_NFW'
            kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                 'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                 'center_y': kwargs_lens['center_y'], 'r200': self.kwargs_lens_clump_init['r200']}
        elif clump == 'SIS':
            self.kwargs_options['lens_type'] = 'SPEP_SIS'
            kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                 'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                 'center_y': kwargs_lens['center_y']}
        elif clump == 'SPP':
            self.kwargs_options['lens_type'] = 'SPEP_SPP'
            kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                 'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                 'center_y': kwargs_lens['center_y']}
        elif clump == 'SPP_SHAPELETS':
            self.kwargs_options['lens_type'] = 'SPEP_SPP_SHAPELETS'
            kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q']
                , 'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                 'center_y': kwargs_lens['center_y']
                , 'coeffs': kwargs_lens['coeffs'], 'beta': kwargs_lens['beta']}
        else:
            raise ValueError('Clump model not valid!')
        # this are the parameters which are held constant while sampling
        kwargs_fixed_source = {'I0_sersic': 1, 'center_x': 0, 'center_y': 0}
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = kwargs_else

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        if self.kwargs_options.get('ML_prior', False) is True:
            # only works for SPP clump
            self.kwargs_lens_clump_init['center_x_spp'] = kwargs_lens_light['center_x_2']
            self.kwargs_lens_clump_init['center_y_spp'] = kwargs_lens_light['center_y_2']
            self.kwargs_lens_clump_sigma_init['center_x_spp_sigma'] = 0.1
            self.kwargs_lens_clump_sigma_init['center_x_spp_sigma'] = 0.1
        kwargs_mean_lens = dict(kwargs_lens.items() + self.kwargs_lens_clump_init.items())
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = {}
        # sigma values
        kwargs_sigma_lens = dict(self.kwargs_lens_sigma_weak.items() + self.kwargs_lens_clump_sigma_init.items())
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = {}
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Perturbed')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_suyu(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                        num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        """
        finds a lens configuration such that the catalogue is matched with a given subclump configuration
        """
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['solver_type'] = self.kwargs_options['ellipse_type']
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        self.kwargs_options['fix_subclump'] = True

        self.kwargs_options['lens_type'] = 'SPEP_SIS'
        kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                             'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                             'center_y': kwargs_lens['center_y']
            , 'center_x_sis': kwargs_lens_light['center_x_2'], 'center_y_sis': kwargs_lens_light['center_y_2']}

        # this are the parameters which are held constant while sampling
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values

        kwargs_mean_lens = dict(kwargs_lens.items() + self.kwargs_lens_clump_init.items())
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        # kwargs_mean_else = dict(kwargs_else.items() + {'gamma_ext': self.kwargs_else_init['gamma_ext'], 'psi_ext': self.kwargs_else_init['psi_ext']}.items())
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = dict(self.kwargs_lens_sigma_weak.items() + self.kwargs_lens_clump_sigma_init.items())
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Suyu')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_suyu_add(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                            num_order, num_shapelet_lens, beta_lens, subgrid_res=2, numThreads=1, mpi_monch=False):
        """
        finds a lens configuration such that the catalogue is matched with a given subclump configuration
        """
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['solver_type'] = self.kwargs_options['ellipse_type']
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        self.kwargs_options['fix_subclump'] = True
        self.kwargs_options['num_shapelet_lens'] = num_shapelet_lens

        self.kwargs_options['lens_type'] = 'SPEP_SPP_SHAPELETS'
        kwargs_fixed_lens = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                             'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                             'center_y': kwargs_lens['center_y']
            , 'center_x_spp': self.kwargs_lens_light_init['center_x_2'],
                             'center_y_spp': self.kwargs_lens_light_init['center_y_2'], 'gamma_spp': 2.
            , 'beta': beta_lens, 'center_x_shape': self.kwargs_lens_light_init['center_x'],
                             'center_y_shape': self.kwargs_lens_light_init['center_y']}
        # this are the parameters which are held constant while sampling
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values

        kwargs_mean_lens = dict(
            kwargs_lens.items() + self.kwargs_lens_clump_init.items() + {'coeffs': [0] * num_shapelet_lens}.items())
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        # kwargs_mean_else = dict(kwargs_else.items() + {'gamma_ext': self.kwargs_else_init['gamma_ext'], 'psi_ext': self.kwargs_else_init['psi_ext']}.items())
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = dict(self.kwargs_lens_sigma_init.items() + self.kwargs_lens_clump_sigma_init.items())
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Suyu')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_lens_shapelets(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                  n_iterations, num_order, num_shapelets_lens, beta_lens, subgrid_res=2, numThreads=1,
                                  mpi_monch=False, clump='NONE'):
        """
        find parameter configuration with image comparison only
        :param kwargs_lens:
        :param kwargs_source:
        :param n_particles:
        :param n_iterations:
        :return:
        """
        if clump == 'NONE':
            self.kwargs_options['lens_type'] = 'SPEP_SHAPELETS'
        elif clump == 'NFW':
            self.kwargs_options['lens_type'] = 'SPEP_NFW_SHAPELETS'
        elif clump == 'SIS':
            self.kwargs_options['lens_type'] = 'SPEP_SIS_SHAPELETS'
        elif clump == 'SPP':
            self.kwargs_options['lens_type'] = 'SPEP_SPP_SHAPELETS'
        else:
            raise ValueError('invalide clump parameter:', clump)
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['num_shapelet_lens'] = num_shapelets_lens
        self.kwargs_options['subgrid_res'] = subgrid_res
        # this are the parameters which are held constant while sampling
        kwargs_fixed_shapelets = {'center_x_shape': kwargs_lens['center_x'], 'center_y_shape': kwargs_lens['center_y']}
        kwargs_fixed_lens = dict(kwargs_lens.items() + kwargs_fixed_shapelets.items())
        del kwargs_fixed_lens['gamma']
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = kwargs_else

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_lens_add = {'beta': beta_lens, 'coeffs': [0] * num_shapelets_lens}
        kwargs_sigma_lens_add = {'beta_sigma': self.kwargs_lens_sigma_init['beta_sigma'],
                                 'coeffs_sigma': self.kwargs_lens_sigma_init['coeffs_sigma']}
        kwargs_mean_lens = dict(kwargs_lens.items() + kwargs_lens_add.items())
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = {}
        # sigma values
        kwargs_sigma_lens = dict(self.kwargs_lens_sigma_weak.items() + kwargs_sigma_lens_add.items())
        kwargs_sigma_source = self.kwargs_source_sigma_weak
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_weak
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Shapelet perturb')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_lens_pertrub_shapelets(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                          n_iterations, num_order, subgrid_res=2, numThreads=1, mpi_monch=False,
                                          clump='NONE', dipole=False):
        """
        find parameter configuration with image comparison only
        :param kwargs_lens:
        :param kwargs_source:
        :param n_particles:
        :param n_iterations:
        :return:
        """
        if clump == 'NONE':
            self.kwargs_options['lens_type'] = 'SPEP_SHAPELETS'
        elif clump == 'NFW':
            self.kwargs_options['lens_type'] = 'SPEP_NFW_SHAPELETS'
        elif clump == 'SIS':
            self.kwargs_options['lens_type'] = 'SPEP_SIS_SHAPELETS'
        elif clump == 'SPP':
            if dipole is True:
                self.kwargs_options['lens_type'] = 'SPEP_SPP_DIPOLE_SHAPELETS'
            else:
                self.kwargs_options['lens_type'] = 'SPEP_SPP_SHAPELETS'
        else:
            raise ValueError('invalide clump parameter:', clump)
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['solver_type'] = 'SHAPELETS'
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['num_shapelet_lens'] = 6
        self.kwargs_options['subgrid_res'] = subgrid_res
        # this are the parameters which are held constant while sampling
        kwargs_fixed_shapelets = {'center_x_shape': kwargs_lens['center_x'], 'center_y_shape': kwargs_lens['center_y'],
                                  'coeffs': [0] * 6}
        kwargs_fixed_lens = kwargs_fixed_shapelets
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = kwargs_else

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        beta_lens = max(kwargs_lens['phi_E'].copy(), 0.5)
        if dipole is True:
            kwargs_lens_add = {'beta': beta_lens, 'coupling': self.kwargs_lens_clump_init['coupling']}
            kwargs_sigma_lens_add = {'beta_sigma': self.kwargs_lens_sigma_init['beta_sigma'],
                                     'coupling_sigma': self.kwargs_lens_clump_sigma_init['coupling_sigma'],
                                     'phi_dipole_sigma': self.kwargs_lens_clump_sigma_init['phi_dipole_sigma']}
        else:
            kwargs_lens_add = {'beta': beta_lens}
            kwargs_sigma_lens_add = {'beta_sigma': self.kwargs_lens_sigma_init['beta_sigma']}
        kwargs_mean_lens = dict(kwargs_lens.items() + kwargs_lens_add.items())
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = {}
        # sigma values
        kwargs_sigma_lens = dict(
            self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_weak.items() + kwargs_sigma_lens_add.items())
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = {}
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Shapelet perturb')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_all(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                       num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        """
        finds a lens configuration such that the catalogue is matched with a given subclump configuration
        """
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        # this are the parameters which are held constant while sampling
        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options[
            'lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or \
                        self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            kwargs_fixed_shapelets = {'beta': kwargs_lens['beta'], 'center_x_shape': kwargs_lens['center_x_shape'],
                                      'center_y_shape': kwargs_lens['center_y_shape']}  # , 'coeffs': [0]*6}
        else:
            kwargs_fixed_shapelets = {}
        if self.kwargs_options['solver'] is True and (
                self.kwargs_options['solver_type'] == 'SPEP' or self.kwargs_options['solver_type'] == 'SPEMD'):
            kwargs_fixed_solver = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                   'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                   'center_y': kwargs_lens['center_y']}
        else:
            kwargs_fixed_solver = {}
        kwargs_fixed_add = {}
        if self.kwargs_options.get('fix_subclump', False):
            if 'center_x_sis' in kwargs_lens:
                kwargs_fixed_add = {'center_x_sis': kwargs_lens['center_x_sis'],
                                    'center_y_sis': kwargs_lens['center_y_sis']}
            elif 'center_x_spp' in kwargs_lens:
                kwargs_fixed_add = {'center_x_spp': kwargs_lens['center_x_spp'],
                                    'center_y_spp': kwargs_lens['center_y_spp']}
            elif 'center_x_nfw' in kwargs_lens:
                kwargs_fixed_add = {'center_x_nfw': kwargs_lens['center_x_nfw'],
                                    'center_y_nfw': kwargs_lens['center_y_nfw']}
        kwargs_fixed_lens = dict(
            kwargs_fixed_shapelets.items() + kwargs_fixed_solver.items() + kwargs_fixed_add.items())
        kwargs_fixed_source = {'center_x': kwargs_source['center_x'], 'center_y': kwargs_source['center_y'],
                               'I0_sersic': 1}
        kwargs_fixed_psf = self.kwargs_psf_init
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = self.kwargs_psf_init
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = dict(
            self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_constraint.items())
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Very all')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_external_shear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                  n_iterations, num_order, subgrid_res=2, numThreads=1, mpi_monch=False):
        """
        find parameter configuration with image comparison only
        :param kwargs_lens:
        :param kwargs_source:
        :param n_particles:
        :param n_iterations:
        :return:
        """
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res
        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options[
            'lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or \
                        self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            kwargs_fixed_shapelets = {'beta': kwargs_lens['beta'], 'center_x_shape': kwargs_lens['center_x_shape'],
                                      'center_y_shape': kwargs_lens['center_y_shape']}  # , 'coeffs': [0]*6}
        else:
            kwargs_fixed_shapelets = {}
        if self.kwargs_options['solver'] is True and (
                self.kwargs_options['solver_type'] == 'SPEP' or self.kwargs_options['solver_type'] == 'SPEMD'):
            kwargs_fixed_solver = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                   'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                   'center_y': kwargs_lens['center_y']}
        else:
            kwargs_fixed_solver = {}
        kwargs_fixed_add = {}
        if self.kwargs_options.get('fix_subclump', False):
            if 'center_x_sis' in kwargs_lens:
                kwargs_fixed_add = {'center_x_sis': kwargs_lens['center_x_sis'],
                                    'center_y_sis': kwargs_lens['center_y_sis']}
            elif 'center_x_spp' in kwargs_lens:
                kwargs_fixed_add = {'center_x_spp': kwargs_lens['center_x_spp'],
                                    'center_y_spp': kwargs_lens['center_y_spp']}
            elif 'center_x_nfw' in kwargs_lens:
                kwargs_fixed_add = {'center_x_nfw': kwargs_lens['center_x_nfw'],
                                    'center_y_nfw': kwargs_lens['center_y_nfw']}
        kwargs_fixed_lens = dict(
            kwargs_fixed_shapelets.items() + kwargs_fixed_solver.items() + kwargs_fixed_add.items())
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = self.kwargs_psf_init
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = dict(kwargs_else.items() + {'gamma1': self.kwargs_else_init['gamma1'],
                                                       'gamma2': self.kwargs_else_init['gamma2']}.items())
        # sigma values
        kwargs_sigma_lens = dict(self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_weak.items())
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='External shear')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_add_shapelets(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                                 n_iterations, num_order, num_shapelets_lens, subgrid_res=2, numThreads=1,
                                 mpi_monch=False, clump='NFW'):
        """
        adding additional shapelet coeffs
        """
        if num_shapelets_lens < 6:
            num_shapelets_lens = 6
        self.kwargs_options['X2_type'] = 'image'
        self.kwargs_options['solver'] = True
        self.kwargs_options['solver_type'] = 'SHAPELETS'
        self.kwargs_options['num_shapelet_lens'] = num_shapelets_lens
        self.kwargs_options['fix_source'] = False
        self.kwargs_options['shapelets_off'] = False
        self.kwargs_options['shapelet_order'] = num_order
        self.kwargs_options['subgrid_res'] = subgrid_res

        kwargs_fixed_lens = dict({'beta': kwargs_lens['beta'], 'center_x_shape': kwargs_lens['center_x_shape'],
                                  'center_y_shape': kwargs_lens['center_y_shape']}.items())
        if clump == 'SPP':
            self.kwargs_options['lens_type'] = 'SPEP_SPP_SHAPELETS'
        elif clump == 'NONE':
            self.kwargs_options['lens_type'] = 'SPEP_SHAPELETS'
        else:
            raise ValueError('Clump model not valid!')
        kwargs_fixed_source = {'I0_sersic': 1, 'center_x': 0, 'center_y': 0}
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1, 'center_x_2': kwargs_lens_light['center_x_2'],
                                   'center_y_2': kwargs_lens_light['center_y_2'], 'n_2': kwargs_lens_light['n_2']}
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}
        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        coeffs_new = [0] * num_shapelets_lens
        if 'coeffs' in kwargs_lens:
            coeffs_old = kwargs_lens['coeffs']
            n = len(coeffs_old)
            coeffs_new[0:n] = coeffs_old
        kwargs_mean_lens = dict(kwargs_lens.items() + {'coeffs': coeffs_new}.items())
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        kwargs_sigma_lens_add = {'coeffs_sigma': self.kwargs_lens_sigma_constraint['coeffs_sigma']}
        kwargs_sigma_lens = dict(
            self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_constraint.items() + kwargs_sigma_lens_add.items())
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Perturb shapelets')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_final(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                         subgrid_res=2, numThreads=1, mpi_monch=False, dipole=False):
        """
        varies all possible parameters at once (including the time delay distance)
        """
        self.kwargs_options['subgrid_res'] = subgrid_res
        if 'time_delays' in self.kwargs_data:
            self.kwargs_options['time_delay'] = True
        self.kwargs_options['solver'] = True
        # this are the parameters which are held constant while sampling
        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options[
            'lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or \
                        self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            kwargs_fixed_shapelets = {'beta': kwargs_lens['beta'], 'center_x_shape': kwargs_lens['center_x_shape'],
                                      'center_y_shape': kwargs_lens['center_y_shape']}  # , 'coeffs': [0]*6}
        else:
            kwargs_fixed_shapelets = {}
        if self.kwargs_options['solver'] is True and (
                self.kwargs_options['solver_type'] == 'SPEP' or self.kwargs_options['solver_type'] == 'SPEMD'):
            kwargs_fixed_solver = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                   'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                   'center_y': kwargs_lens['center_y']}
        else:
            kwargs_fixed_solver = {}
        kwargs_fixed_add = {}
        if self.kwargs_options.get('fix_subclump', False):
            if 'center_x_sis' in kwargs_lens:
                kwargs_fixed_add = {'center_x_sis': kwargs_lens['center_x_sis'],
                                    'center_y_sis': kwargs_lens['center_y_sis']}
            elif 'center_x_spp' in kwargs_lens:
                kwargs_fixed_add = {'center_x_spp': kwargs_lens['center_x_spp'],
                                    'center_y_spp': kwargs_lens['center_y_spp'], 'gamma_spp': kwargs_lens['gamma_spp']}
            elif 'center_x_nfw' in kwargs_lens:
                kwargs_fixed_add = {'center_x_nfw': kwargs_lens['center_x_nfw'],
                                    'center_y_nfw': kwargs_lens['center_y_nfw']}
        kwargs_fixed_lens = dict(
            kwargs_fixed_shapelets.items() + kwargs_fixed_solver.items() + kwargs_fixed_add.items())
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = self.kwargs_psf_init
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = dict({'delay_dist': self.kwargs_else_init['delay_dist']}.items() + kwargs_else.items())
        # sigma values
        kwargs_sigma_lens = dict(
            self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_constraint.items())
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = dict(self.kwargs_else_sigma_constraint.items() + {
            'delay_dist_sigma': self.kwargs_else_sigma_init['delay_dist_sigma']}.items())
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='Final')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_subclump(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles, n_iterations,
                      subgrid_res=2, numThreads=1, mpi_monch=False):
        """
        find subclump to addition of the "final" sampling
        """
        self.kwargs_options['subgrid_res'] = subgrid_res
        self.kwargs_options['add_clump'] = True
        self.kwargs_options['fix_source'] = True
        self.kwargs_options['solver'] = True
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        kwargs_fixed_else = dict(kwargs_else.items())

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = {}
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = {'phi_E_clump': self.kwargs_else_init['phi_E_clump'],
                            'r_trunc': self.kwargs_else_init['r_trunc'], 'x_clump': self.kwargs_else_init['x_clump'],
                            'y_clump': self.kwargs_else_init['y_clump']}
        # sigma values
        kwargs_sigma_lens = {}
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='sub-clump detection')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def find_param_beta_source(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_particles,
                               n_iterations, numThreads=1, mpi_monch=False):
        """
        find subclump to addition of the "final" sampling
        """
        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        kwargs_fixed_lens_light = kwargs_lens_light
        shapelet_beta = kwargs_else['shapelet_beta']
        del kwargs_else['shapelet_beta']
        kwargs_fixed_else = kwargs_else

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = {}
        kwargs_mean_source = {}
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = {}
        kwargs_mean_else = {'shapelet_beta': shapelet_beta}
        # sigma values
        kwargs_sigma_lens = {}
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = {}
        kwargs_sigma_else = self.kwargs_else_sigma_init
        lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
            kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
            kwargs_fixed_psf, kwargs_mean_psf, kwargs_sigma_psf,
            kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
            kwargs_fixed_else, kwargs_mean_else, kwargs_sigma_else,
            threadCount=numThreads, mpi_monch=mpi_monch, print_key='shapelet beta iteration')
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain, param_list

    def mcmc_arc(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_burn, n_run, walkerRatio,
                 numThreads=1, mpi_monch=False, init_positions=None, fix_lens_light=False):
        """
        MCMC
        """

        # this are the parameters which are held constant while sampling
        kwargs_fixed_lens = {}
        # del kwargs_fixed_lens['gamma']
        kwargs_fixed_source = {'center_x': kwargs_source['center_x'], 'center_y': kwargs_source['center_y'],
                               'I0_sersic': 1}
        kwargs_fixed_psf = self.kwargs_psf_init
        if fix_lens_light is True:
            kwargs_fixed_lens_light = kwargs_lens_light
        else:
            if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
                kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
            elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
                'lens_light_type'] == 'SERSIC_ELLIPSE':
                kwargs_fixed_lens_light = {'I0_sersic': 1}
            else:
                kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = {}
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = self.kwargs_lens_sigma_constraint
        kwargs_sigma_source = self.kwargs_source_sigma_constraint
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        kwargs_prior_lens = dict(kwargs_mean_lens.items() + kwargs_sigma_lens.items())
        kwargs_prior_source = dict(kwargs_mean_source.items() + kwargs_sigma_source.items())
        kwargs_prior_psf = dict(kwargs_mean_psf.items() + kwargs_sigma_psf.items())
        kwargs_prior_lens_light = dict(kwargs_mean_lens_light.items() + kwargs_sigma_lens_light.items())
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())

        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, psf_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_psf_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_source = dict(kwargs_fixed_source.items() + source_fix.items())
        kwargs_fixed_psf = dict(kwargs_fixed_psf.items() + psf_fix.items())
        kwargs_fixed_lens_light = dict(kwargs_fixed_lens_light.items() + lens_light_fix.items())
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())

        # initialise mcmc classes
        mcmc_class = MCMC_sampler(self.kwargs_data, self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                                  kwargs_fixed_psf, kwargs_fixed_lens_light, kwargs_fixed_else)
        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source, kwargs_prior_psf,
                                                         kwargs_prior_lens_light, kwargs_prior_else)
        num_param, param_list = param_class.num_param()

        # run MCMC
        if not init_positions is None:
            initpos = ReusePositionGenerator(init_positions)
        else:
            initpos = None

        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=numThreads,
                                           mpi_monch=mpi_monch, init_pos=initpos)
        return samples, param_list, dist

    def mcmc_run(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_burn, n_run, walkerRatio,
                 numThreads=1, mpi_monch=False, dipole=False, init_positions=None, fix_lens_light=True):
        """
        MCMC
        """
        # self.kwargs_options['time_delay'] = True
        # this are the parameters which are held constant while sampling
        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options[
            'lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or \
                        self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            kwargs_fixed_shapelets = {'beta': kwargs_lens['beta'], 'center_x_shape': kwargs_lens['center_x_shape'],
                                      'center_y_shape': kwargs_lens['center_y_shape']}
        else:
            kwargs_fixed_shapelets = {}
        if self.kwargs_options['solver'] is True and (
                self.kwargs_options['solver_type'] == 'SPEP' or self.kwargs_options['solver_type'] == 'SPEMD'):
            kwargs_fixed_solver = {'phi_E': kwargs_lens['phi_E'], 'q': kwargs_lens['q'],
                                   'phi_G': kwargs_lens['phi_G'], 'center_x': kwargs_lens['center_x'],
                                   'center_y': kwargs_lens['center_y']}
        else:
            kwargs_fixed_solver = {}
        kwargs_fixed_add = {}
        if self.kwargs_options.get('fix_subclump', False) is True:
            if 'center_x_sis' in kwargs_lens:
                kwargs_fixed_add = {'center_x_sis': kwargs_lens['center_x_sis'],
                                    'center_y_sis': kwargs_lens['center_y_sis']}
            elif 'center_x_spp' in kwargs_lens:
                kwargs_fixed_add = {'center_x_spp': kwargs_lens['center_x_spp'],
                                    'center_y_spp': kwargs_lens['center_y_spp'], 'gamma_spp': kwargs_lens['gamma_spp']}
            elif 'center_x_nfw' in kwargs_lens:
                kwargs_fixed_add = {'center_x_nfw': kwargs_lens['center_x_nfw'],
                                    'center_y_nfw': kwargs_lens['center_y_nfw']}
        kwargs_fixed_lens = dict(
            kwargs_fixed_shapelets.items() + kwargs_fixed_solver.items() + kwargs_fixed_add.items())
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        if fix_lens_light:
            kwargs_fixed_lens_light = kwargs_lens_light
        else:
            if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
                kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
            elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
                'lens_light_type'] == 'SERSIC_ELLIPSE':
                kwargs_fixed_lens_light = {'I0_sersic': 1}
            else:
                kwargs_fixed_lens_light = {}
        kwargs_fixed_else = {'ra_pos': kwargs_else['ra_pos'], 'dec_pos': kwargs_else['dec_pos'],
                             'shapelet_beta': kwargs_else['shapelet_beta']}

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = self.kwargs_psf_init
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = dict(
            self.kwargs_lens_sigma_constraint.items() + self.kwargs_lens_clump_sigma_constraint.items())
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = self.kwargs_else_sigma_constraint
        kwargs_prior_lens = dict(kwargs_mean_lens.items() + kwargs_sigma_lens.items())
        kwargs_prior_source = dict(kwargs_mean_source.items() + kwargs_sigma_source.items())
        kwargs_prior_psf = dict(kwargs_mean_psf.items() + kwargs_sigma_psf.items())
        kwargs_prior_lens_light = dict(kwargs_mean_lens_light.items() + kwargs_sigma_lens_light.items())
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())
        # initialise mcmc classes
        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, psf_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_psf_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_source = dict(kwargs_fixed_source.items() + source_fix.items())
        kwargs_fixed_psf = dict(kwargs_fixed_psf.items() + psf_fix.items())
        kwargs_fixed_lens_light = dict(kwargs_fixed_lens_light.items() + lens_light_fix.items())
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())

        mcmc_class = MCMC_sampler(self.kwargs_data, self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                                  kwargs_fixed_psf, kwargs_fixed_lens_light, kwargs_fixed_else)
        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source, kwargs_prior_psf,
                                                         kwargs_prior_lens_light, kwargs_prior_else)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_positions is None:
            initpos = ReusePositionGenerator(init_positions)
        else:
            initpos = None

        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=numThreads,
                                           mpi_monch=mpi_monch, init_pos=initpos)
        return samples, param_list, dist

    def mcmc_lens_light(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, n_run, n_burn, walkerRatio,
                        subgrid_res=2, numThreads=1, mpi_monch=False, dipole=False, init_positions=None):
        """
        MCMC for the lens light parameters only
        """
        self.kwargs_options['subgrid_res'] = subgrid_res
        self.kwargs_options['solver'] = False
        self.kwargs_options['fix_source'] = True
        # this are the parameters which are held constant while sampling

        kwargs_fixed_lens = kwargs_lens
        kwargs_fixed_source = kwargs_source
        kwargs_fixed_psf = self.kwargs_psf_init
        if self.kwargs_options['lens_light_type'] == 'TRIPLE_SERSIC':
            kwargs_fixed_lens_light = {'I0_sersic': 1, 'I0_2': 1, 'I0_3': 1}
        elif self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options[
            'lens_light_type'] == 'SERSIC_ELLIPSE':
            kwargs_fixed_lens_light = {'I0_sersic': 1}
        else:
            kwargs_fixed_lens_light = {}
        kwargs_fixed_else = kwargs_else

        # mean and sigma of the starting walkers (only for varying parameters)
        # mean values
        kwargs_mean_lens = kwargs_lens
        kwargs_mean_source = kwargs_source
        kwargs_mean_psf = self.kwargs_psf_init
        kwargs_mean_lens_light = kwargs_lens_light
        kwargs_mean_else = kwargs_else
        # sigma values
        kwargs_sigma_lens = {}
        kwargs_sigma_source = {}
        kwargs_sigma_psf = {}
        kwargs_sigma_lens_light = self.kwargs_lens_light_sigma_constraint
        kwargs_sigma_else = {}
        kwargs_prior_lens = dict(kwargs_mean_lens.items() + kwargs_sigma_lens.items())
        kwargs_prior_source = dict(kwargs_mean_source.items() + kwargs_sigma_source.items())
        kwargs_prior_psf = dict(kwargs_mean_psf.items() + kwargs_sigma_psf.items())
        kwargs_prior_lens_light = dict(kwargs_mean_lens_light.items() + kwargs_sigma_lens_light.items())
        kwargs_prior_else = dict(kwargs_mean_else.items() + kwargs_sigma_else.items())

        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        lens_fix, source_fix, psf_fix, lens_light_fix, else_fix = param_class.add_to_fixed(self.kwargs_lens_fixed,
                                                                                           self.kwargs_source_fixed,
                                                                                           self.kwargs_psf_fixed,
                                                                                           self.kwargs_lens_light_fixed,
                                                                                           self.kwargs_else_fixed)
        kwargs_fixed_lens = dict(kwargs_fixed_lens.items() + lens_fix.items())
        kwargs_fixed_source = dict(kwargs_fixed_source.items() + source_fix.items())
        kwargs_fixed_psf = dict(kwargs_fixed_psf.items() + psf_fix.items())
        kwargs_fixed_lens_light = dict(kwargs_fixed_lens_light.items() + lens_light_fix.items())
        kwargs_fixed_else = dict(kwargs_fixed_else.items() + else_fix.items())
        # initialise mcmc classes
        mcmc_class = MCMC_sampler(self.kwargs_data, self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source,
                                  kwargs_fixed_psf, kwargs_fixed_lens_light, kwargs_fixed_else)
        param_class = Param(self.kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf,
                            kwargs_fixed_lens_light, kwargs_fixed_else)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source, kwargs_prior_psf,
                                                         kwargs_prior_lens_light, kwargs_prior_else)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_positions is None:
            initpos = ReusePositionGenerator(init_positions)
        else:
            initpos = None

        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=numThreads,
                                           mpi_monch=mpi_monch, init_pos=initpos)
        return samples, param_list, dist