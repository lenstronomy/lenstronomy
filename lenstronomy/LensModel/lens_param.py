import numpy as np
import lenstronomy.Util.param_util as param_util


class LensParam(object):
    """
    class to handle the lens model parameter
    """
    def __init__(self, lens_model_list, kwargs_fixed, num_images=0, solver_type='NONE', num_shapelet_lens=0,
                 mass_scaling=False):
        """

        :param kwargs_options:
        :param kwargs_fixed:
        """
        self.model_list = lens_model_list
        self.kwargs_fixed = kwargs_fixed
        self._num_images = num_images
        self._solver_type = solver_type
        self._num_shapelet_lens = num_shapelet_lens

    def getParams(self, args, i):
        kwargs_list = []
        for k, model in enumerate(self.model_list):
            kwargs = {}
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['SHEAR', 'FOREGROUND_SHEAR']:
                if not 'e1' in kwargs_fixed:
                    kwargs['e1'] = args[i]
                    i += 1
                else:
                    kwargs['e1'] = kwargs_fixed['e1']
                if not 'e2' in kwargs_fixed:
                    kwargs['e2'] = args[i]
                    i += 1
                else:
                    kwargs['e2'] = kwargs_fixed['e2']
            if model == 'CONVERGENCE':
                if not 'kappa_ext' in kwargs_fixed:
                    kwargs['kappa_ext'] = args[i]
                    i += 1
                else:
                    kwargs['kappa_ext'] = kwargs_fixed['kappa_ext']
            if model == 'FLEXION':
                if not 'g1' in kwargs_fixed:
                    kwargs['g1'] = args[i]
                    i += 1
                else:
                    kwargs['g1'] = kwargs_fixed['g1']
                if not 'g2' in kwargs_fixed:
                    kwargs['g2'] = args[i]
                    i += 1
                else:
                    kwargs['g2'] = kwargs_fixed['g2']
                if not 'g3' in kwargs_fixed:
                    kwargs['g3'] = args[i]
                    i += 1
                else:
                    kwargs['g3'] = kwargs_fixed['g3']
                if not 'g4' in kwargs_fixed:
                    kwargs['g4'] = args[i]
                    i += 1
                else:
                    kwargs['g4'] = kwargs_fixed['g4']
            if model in ['GAUSSIAN', 'GAUSSIAN_KAPPA']:
                if not 'amp' in kwargs_fixed:
                    kwargs['amp'] = args[i]
                    i += 1
                else:
                    kwargs['amp'] = kwargs_fixed['amp']
            if model in ['GAUSSIAN']:
                if not 'sigma_x' in kwargs_fixed:
                    kwargs['sigma_x'] = args[i]
                    i += 1
                else:
                    kwargs['sigma_x'] = kwargs_fixed['sigma_x']
                if not 'sigma_y' in kwargs_fixed:
                    kwargs['sigma_y'] = args[i]
                    i += 1
                else:
                    kwargs['sigma_y'] = kwargs_fixed['sigma_y']
            if model in ['GAUSSIAN_KAPPA']:
                if not 'sigma' in kwargs_fixed:
                    kwargs['sigma'] = args[i]
                    i += 1
                else:
                    kwargs['sigma'] = kwargs_fixed['sigma']

            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIS', 'SIS_TRUNCATED', 'SPP', 'SIE']:
                if not 'theta_E' in kwargs_fixed:
                    kwargs['theta_E'] = args[i]
                    i += 1
                else:
                    kwargs['theta_E'] = kwargs_fixed['theta_E']
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SPP']:
                if not 'gamma' in kwargs_fixed:
                    kwargs['gamma'] = args[i]
                    i += 1
                else:
                    kwargs['gamma'] = kwargs_fixed['gamma']
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'NFW_ELLIPSE', 'SERSIC_ELLIPSE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE', 'SIE']:
                if not 'e1' in kwargs_fixed:
                    kwargs['e1'] = args[i]
                    i += 1
                else:
                    kwargs['e1'] = kwargs_fixed['e1']
                if not 'e2' in kwargs_fixed:
                    kwargs['e2'] = args[i]
                    i += 1
                else:
                    kwargs['e2'] = kwargs_fixed['e2']

            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'Rs' in kwargs_fixed:
                    kwargs['Rs'] = args[i]
                    i += 1
                else:
                    kwargs['Rs'] = kwargs_fixed['Rs']
            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'theta_Rs' in kwargs_fixed:
                    kwargs['theta_Rs'] = args[i]
                    i += 1
                else:
                    kwargs['theta_Rs'] = kwargs_fixed['theta_Rs']
            if model in ['TNFW', 'SIS_TRUNCATED']:
                if not 'r_trunc' in kwargs_fixed:
                    kwargs['r_trunc'] = args[i]
                    i += 1
                else:
                    kwargs['r_trunc'] = kwargs_fixed['r_trunc']

            if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART']:
                if not 'beta' in kwargs_fixed:
                    kwargs['beta'] = args[i]
                    i += 1
                else:
                    kwargs['beta'] = kwargs_fixed['beta']
                if not 'coeffs' in kwargs_fixed:
                    num_coeffs = self._num_shapelet_lens
                    if self._solver_type == 'SHAPELETS':
                        if self._num_images == 4:
                            num_coeffs -= 6
                            coeffs = args[i:i+num_coeffs]
                            coeffs = [0,0,0,0,0,0] + list(coeffs[0:])
                        elif self._num_images == 2:
                            num_coeffs -=3
                            coeffs = args[i:i+num_coeffs]
                            coeffs = [0, 0, 0] + list(coeffs[0:])
                        else:
                            raise ValueError("Option for solver_type not valid!")
                        kwargs['coeffs'] = coeffs
                    else:
                        kwargs['coeffs'] = args[i:i+num_coeffs]
                    i += num_coeffs
                else:
                    kwargs['coeffs'] = kwargs_fixed['coeffs']

            if model in ['DIPOLE']:
                if not 'coupling' in kwargs_fixed:
                    kwargs['coupling'] = args[i]
                    i += 1
                else:
                    kwargs['coupling'] = kwargs_fixed['coupling']
                if not 'phi_dipole' in kwargs_fixed and self.kwargs_options.get('phi_dipole_decoupling', False) is True:
                    kwargs['phi_dipole'] = args[i]
                    i += 1
                else:
                    print(kwargs_fixed, 'test')
                    kwargs['phi_dipole'] = kwargs_fixed['phi_dipole']
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'n_sersic' in kwargs_fixed:
                    kwargs['n_sersic'] = args[i]
                    i += 1
                else:
                    kwargs['n_sersic'] = kwargs_fixed['n_sersic']
                if not 'R_sersic' in kwargs_fixed:
                    kwargs['R_sersic'] = args[i]
                    i += 1
                else:
                    kwargs['R_sersic'] = kwargs_fixed['R_sersic']
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'k_eff' in kwargs_fixed:
                    kwargs['k_eff'] = args[i]
                    i += 1
                else:
                    kwargs['k_eff'] = kwargs_fixed['k_eff']
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    kwargs['sigma0'] = args[i]
                    i += 1
                else:
                    kwargs['sigma0'] = kwargs_fixed['sigma0']
                if not 'Rs' in kwargs_fixed:
                    kwargs['Rs'] = args[i]
                    i += 1
                else:
                    kwargs['Rs'] = kwargs_fixed['Rs']
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    kwargs['Ra'] = args[i]
                    i += 1
                else:
                    kwargs['Ra'] = kwargs_fixed['Ra']
            if model in ['SPEMD_SMOOTH']:
                if not 's_scale' in kwargs_fixed:
                    kwargs['s_scale'] = args[i]
                    i += 1
            if model in ['SIS', 'SIE', 'SPP', 'SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'NFW', 'TNFW', 'NFW_ELLIPSE', 'SIS_TRUNCATED', 'SHAPELETS_POLAR',
                         'SHAPELETS_CART', 'DIPOLE', 'GAUSSIAN', 'GAUSSIAN_KAPPA', 'SERSIC', 'SERSIC_ELLIPSE', 'HERNQUIST',
                         'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'center_x' in kwargs_fixed:
                    kwargs['center_x'] = args[i]
                    i += 1
                else:
                    kwargs['center_x'] = kwargs_fixed['center_x']
                if not 'center_y' in kwargs_fixed:
                    kwargs['center_y'] = args[i]
                    i += 1
                else:
                    kwargs['center_y'] = kwargs_fixed['center_y']
            if model in ['INTERPOL', 'INTERPOL_SCALED']:
                #grid_interp_x = None, grid_interp_y = None, f_ = None, f_x = None, f_y = None, f_xx = None, f_yy = None, f_xy = None
                kwargs['grid_interp_x'] = kwargs_fixed['grid_interp_x']
                kwargs['grid_interp_y'] = kwargs_fixed['grid_interp_y']
                kwargs['f_x'] = kwargs_fixed['f_x']
                kwargs['f_y'] = kwargs_fixed['f_y']
            if model in ['INTERPOL_SCALED']:
                if not 'scale_factor' in kwargs_fixed:
                    kwargs['scale_factor'] = args[i]
                    i += 1
                else:
                    kwargs['scale_factor'] = kwargs_fixed['scale_factor']
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def setParams(self, kwargs_list, bounds=None):
        """

        :param kwargs:
        :return:
        """
        args = []
        for k, model in enumerate(self.model_list):
            kwargs = kwargs_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['SHEAR', 'FOREGROUND_SHEAR']:
                if not 'e1' in kwargs_fixed:
                    args.append(kwargs['e1'])
                if not 'e2' in kwargs_fixed:
                    args.append(kwargs['e2'])
            if model == 'CONVERGENCE':
                if not 'kappa_ext' in kwargs_fixed:
                    args.append(kwargs['kappa_ext'])
            if model == 'FLEXION':
                if not 'g1' in kwargs_fixed:
                    args.append(kwargs['g1'])
                if not 'g2' in kwargs_fixed:
                    args.append(kwargs['g2'])
                if not 'g3' in kwargs_fixed:
                    args.append(kwargs['g3'])
                if not 'g4' in kwargs_fixed:
                    args.append(kwargs['g4'])
            if model in ['GAUSSIAN', 'GAUSSIAN_KAPPA']:
                if not 'amp' in kwargs_fixed:
                    args.append(kwargs['amp'])
            if model in ['GAUSSIAN']:
                if not 'sigma_x' in kwargs_fixed:
                    args.append(kwargs['sigma_x'])
                if not 'sigma_y' in kwargs_fixed:
                    args.append(kwargs['sigma_y'])
            if model in ['GAUSSIAN_KAPPA']:
                if not 'sigma' in kwargs_fixed:
                    args.append(kwargs['sigma'])
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIS', 'SIE', 'SIS_TRUNCATED', 'SPP']:
                if not 'theta_E' in kwargs_fixed:
                    args.append(kwargs['theta_E'])
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SPP']:
                if not 'gamma' in kwargs_fixed:
                    args.append(kwargs['gamma'])
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIE', 'NFW_ELLIPSE', 'SERSIC_ELLIPSE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    args.append(kwargs['e1'])
                if not 'e2' in kwargs_fixed:
                    args.append(kwargs['e2'])
            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'Rs' in kwargs_fixed:
                    args.append(kwargs['Rs'])
            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'theta_Rs' in kwargs_fixed:
                    args.append(kwargs['theta_Rs'])
            if model in ['TNFW', 'SIS_TRUNCATED']:
                if not 'r_trunc' in kwargs_fixed:
                    args.append(kwargs['r_trunc'])

            if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART']:
                if not 'beta' in kwargs_fixed:
                    args.append(kwargs['beta'])
                if not 'coeffs' in kwargs_fixed:
                    coeffs = kwargs['coeffs']
                    if self._solver_type == 'SHAPELETS':
                        if self._num_images == 4:
                            coeffs = coeffs[6:]
                        elif self._num_images == 2:
                            coeffs = coeffs[3:]
                    args += list(coeffs)

            if model in ['DIPOLE']:
                if not 'coupling' in kwargs_fixed:
                    args.append(kwargs['coupling'])
                if not 'phi_dipole' in kwargs_fixed and self.kwargs_options.get('phi_dipole_decoupling', False) is True:
                    args.append(kwargs['phi_dipole'])
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'n_sersic' in kwargs_fixed:
                    args.append(kwargs['n_sersic'])
                if not 'R_sersic' in kwargs_fixed:
                    args.append(kwargs['R_sersic'])
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'k_eff' in kwargs_fixed:
                    args.append(kwargs['k_eff'])
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    args.append(kwargs['sigma0'])
                if not 'Rs' in kwargs_fixed:
                    args.append(kwargs['Rs'])
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    args.append(kwargs['Ra'])
            if model in ['SPEMD_SMOOTH']:
                if not 's_scale' in kwargs_fixed:
                    args.append(kwargs['s_scale'])
            if model in ['SIS', 'SIE', 'SPP', 'SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'NFW', 'TNFW', 'NFW_ELLIPSE', 'SIS_TRUNCATED', 'SHAPELETS_POLAR',
                                 'SHAPELETS_CART', 'DIPOLE', 'GAUSSIAN', 'GAUSSIAN_KAPPA', 'SERSIC', 'SERSIC_ELLIPSE',
                         'HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'center_x' in kwargs_fixed:
                    args.append(kwargs['center_x'])
                if not 'center_y' in kwargs_fixed:
                    args.append(kwargs['center_y'])
            if model in ['INTERPOL_SCALED']:
                if not 'scale_factor' in kwargs_fixed:
                    args.append(kwargs['scale_factor'])
        return args

    def param_init(self, kwargs_mean_list):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        for k, model in enumerate(self.model_list):
            kwargs_mean = kwargs_mean_list[k]
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['SHEAR', 'FOREGROUND_SHEAR']:
                if not 'e1' in kwargs_fixed:
                    mean.append(kwargs_mean['e1'])
                    sigma.append(kwargs_mean['shear_sigma'])
                if not 'e2' in kwargs_fixed:
                    mean.append(kwargs_mean['e2'])
                    sigma.append(kwargs_mean['shear_sigma'])
            if model == 'CONVERGENCE':
                if not 'kappa_ext' in kwargs_fixed:
                    mean.append(kwargs_mean['kappa_ext'])
                    sigma.append(kwargs_mean['kappa_ext_sigma'])
            if model == 'FLEXION':
                if not 'g1' in kwargs_fixed:
                    mean.append(kwargs_mean['g1'])
                    sigma.append(kwargs_mean['flexion_sigma'])
                if not 'g2' in kwargs_fixed:
                    mean.append(kwargs_mean['g2'])
                    sigma.append(kwargs_mean['flexion_sigma'])
                if not 'g3' in kwargs_fixed:
                    mean.append(kwargs_mean['g3'])
                    sigma.append(kwargs_mean['flexion_sigma'])
                if not 'g4' in kwargs_fixed:
                    mean.append(kwargs_mean['g4'])
                    sigma.append(kwargs_mean['flexion_sigma'])
            if model in ['GAUSSIAN', 'GAUSSIAN_KAPPA']:
                if not 'amp' in kwargs_fixed:
                    mean.append(kwargs_mean['amp'])
                    sigma.append(kwargs_mean['amp_sigma'])
            if model in ['GAUSSIAN']:
                if not 'sigma_x' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma_x'])
                    sigma.append(kwargs_mean['sigma_x_sigma'])
                if not 'sigma_y' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma_y'])
                    sigma.append(kwargs_mean['sigma_y_sigma'])
            if model in ['GAUSSIAN_KAPPA']:
                if not 'sigma' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma'])
                    sigma.append(kwargs_mean['sigma_sigma'])

            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIS', 'SIE', 'SIS_TRUNCATED', 'SPP']:
                if not 'theta_E' in kwargs_fixed:
                    mean.append(kwargs_mean['theta_E'])
                    sigma.append(kwargs_mean['theta_E_sigma'])
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SPP']:
                if not 'gamma' in kwargs_fixed:
                    mean.append(kwargs_mean['gamma'])
                    sigma.append(kwargs_mean['gamma_sigma'])
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIE', 'NFW_ELLIPSE', 'SERSIC_ELLIPSE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    mean.append(kwargs_mean['e1'])
                    sigma.append(kwargs_mean['ellipse_sigma'])
                if not 'e2' in kwargs_fixed:
                    mean.append(kwargs_mean['e2'])
                    sigma.append(kwargs_mean['ellipse_sigma'])

            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'Rs' in kwargs_fixed:
                    mean.append(kwargs_mean['Rs'])
                    sigma.append(kwargs_mean['Rs_sigma'])
            if model in ['NFW', 'TNFW', 'NFW_ELLIPSE']:
                if not 'theta_Rs' in kwargs_fixed:
                    mean.append(kwargs_mean['theta_Rs'])
                    sigma.append(kwargs_mean['theta_Rs_sigma'])
            if model in ['TNFW', 'SIS_TRUNCATED']:
                if not 'r_trunc' in kwargs_fixed:
                    mean.append(kwargs_mean['r_trunc'])
                    sigma.append(kwargs_mean['r_trunc_sigma'])

            if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART']:
                if not 'beta' in kwargs_fixed:
                    mean.append(kwargs_mean['beta'])
                    sigma.append(kwargs_mean['beta_sigma'])
                if not 'coeffs' in kwargs_fixed:
                    coeffs = kwargs_mean['coeffs']
                    if self._solver_type == 'SHAPELETS':
                        if self._num_images == 4:
                            coeffs = coeffs[6:]
                        elif self._num_images == 2:
                            coeffs = coeffs[3:]
                    for i in range(0, len(coeffs)):
                        mean.append(coeffs[i])
                        sigma.append(kwargs_mean['coeffs_sigma'])

            if model in ['DIPOLE']:
                if not 'coupling' in kwargs_fixed:
                    mean.append(kwargs_mean['coupling'])
                    sigma.append(kwargs_mean['coupling_sigma'])
                if not 'phi_dipole' in kwargs_fixed and self.kwargs_options.get('phi_dipole_decoupling', False) is True:
                    mean.append(kwargs_mean['phi_dipole'])
                    sigma.append(kwargs_mean['phi_dipole_sigma'])
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'n_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['n_sersic'])
                    sigma.append(kwargs_mean['n_sersic_sigma'])
                if not 'R_sersic' in kwargs_fixed:
                    mean.append(kwargs_mean['R_sersic'])
                    sigma.append(kwargs_mean['R_sersic_sigma'])
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'k_eff' in kwargs_fixed:
                    mean.append(kwargs_mean['k_eff'])
                    sigma.append(kwargs_mean['k_eff_sigma'])
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    mean.append(kwargs_mean['sigma0'])
                    sigma.append(kwargs_mean['sigma0_sigma'])
                if not 'Rs' in kwargs_fixed:
                    mean.append(kwargs_mean['Rs'])
                    sigma.append(kwargs_mean['Rs_sigma'])
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    mean.append(kwargs_mean['Ra'])
                    sigma.append(kwargs_mean['Ra_sigma'])
            if model in ['SPEMD_SMOOTH']:
                if not 's_scale' in kwargs_fixed:
                    mean.append(kwargs_mean['s_scale'])
                    sigma.append(kwargs_mean['s_scale_sigma'])
            if model in ['SIS', 'SIE', 'SPP', 'SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'NFW', 'TNFW', 'NFW_ELLIPSE', 'SIS_TRUNCATED', 'SHAPELETS_POLAR'
                , 'SHAPELETS_CART', 'DIPOLE', 'GAUSSIAN', 'GAUSSIAN_KAPPA', 'SERSIC', 'SERSIC_ELLIPSE', 'HERNQUIST',
                         'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'center_x' in kwargs_fixed:
                    mean.append(kwargs_mean['center_x'])
                    sigma.append(kwargs_mean['center_x_sigma'])
                if not 'center_y' in kwargs_fixed:
                    mean.append(kwargs_mean['center_y'])
                    sigma.append(kwargs_mean['center_y_sigma'])
            if model in ['INTERPOL_SCALED']:
                if not 'scale_factor' in kwargs_fixed:
                    mean.append(kwargs_mean['scale_factor'])
                    sigma.append(kwargs_mean['scale_factor_sigma'])
        return mean, sigma

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        for k, model in enumerate(self.model_list):
            kwargs_fixed = self.kwargs_fixed[k]
            if model in ['SHEAR', 'FOREGROUND_SHEAR']:
                if not 'e1' in kwargs_fixed:
                    num += 1
                    list.append('e1')
                if not 'e2' in kwargs_fixed:
                    num += 1
                    list.append('e2')
            if model == 'CONVERGENCE':
                if not 'kappa_ext' in kwargs_fixed:
                    num += 1
                    list.append('kappa_ext')
            if model == 'FLEXION':
                if not 'g1' in kwargs_fixed:
                    num += 1
                    list.append('g1')
                if not 'g2' in kwargs_fixed:
                    num += 1
                    list.append('g2')
                if not 'g3' in kwargs_fixed:
                    num += 1
                    list.append('g3')
                if not 'g4' in kwargs_fixed:
                    num += 1
                    list.append('g4')
            if model in ['GAUSSIAN', 'GAUSSIAN_KAPPA']:
                if not 'amp' in kwargs_fixed:
                    num += 1
                    list.append('amp_lens')
            if model in ['GAUSSIAN']:
                if not 'sigma_x' in kwargs_fixed:
                    num += 1
                    list.append('sigma_x_lens')
                if not 'sigma_y' in kwargs_fixed:
                    num += 1
                    list.append('sigma_y_lens')
            if model in ['GAUSSIAN_KAPPA']:
                if not 'sigma' in kwargs_fixed:
                    num += 1
                    list.append('sigma_lens')
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIS', 'SIE', 'SIS_TRUNCATED', 'SPP']:
                if not 'theta_E' in kwargs_fixed:
                    num += 1
                    list.append('theta_E')
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SPP']:
                if not 'gamma' in kwargs_fixed:
                    num += 1
                    list.append('gamma_lens')
            if model in ['SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'SIE', 'NFW_ELLIPSE', 'SERSIC_ELLIPSE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE']:
                if not 'e1' in kwargs_fixed:
                    num += 1
                    list.append('e1_lens')
                if not 'e2' in kwargs_fixed:
                    num += 1
                    list.append('e2_lens')
            if model in ['NFW', 'NFW_ELLIPSE', 'TNFW']:
                if not 'Rs' in kwargs_fixed:
                    num += 1
                    list.append('Rs_nfw')
            if model in ['NFW', 'NFW_ELLIPSE', 'TNFW']:
                if not 'theta_Rs' in kwargs_fixed:
                    num += 1
                    list.append('theta_Rs_nfw')
            if model in ['TNFW', 'SIS_TRUNCATED']:
                if not 'r_trunc' in kwargs_fixed:
                    num += 1
                    list.append('r_trunc')

            if model in ['SHAPELETS_POLAR', 'SHAPELETS_CART']:
                if not 'beta' in kwargs_fixed:
                    num += 1
                    list.append('beta_lens')
                if not 'coeffs' in kwargs_fixed:
                    num_coeffs = self._num_shapelet_lens
                    if self._solver_type == 'SHAPELETS':
                        if self._num_images == 4:
                            num_coeffs -= 6
                        elif self._num_images == 2:
                            num_coeffs -= 3
                    num += num_coeffs
                    list += ['coeff']*num_coeffs

            if model in ['DIPOLE']:
                if not 'coupling' in kwargs_fixed:
                    num += 1
                    list.append('coupling')
                if not 'phi_dipole' in kwargs_fixed and self.kwargs_options.get('phi_dipole_decoupling', False) is True:
                    num += 1
                    list.append('phi_dipole')
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'n_sersic' in kwargs_fixed:
                    num += 1
                    list.append('n_sersic_lens')
                if not 'R_sersic' in kwargs_fixed:
                    num += 1
                    list.append('R_sersic_lens')
            if model in ['SERSIC', 'SERSIC_ELLIPSE']:
                if not 'k_eff' in kwargs_fixed:
                    num += 1
                    list.append('k_eff_lens')
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'sigma0' in kwargs_fixed:
                    list.append('sigma0_')
                    num += 1
                if not 'Rs' in kwargs_fixed:
                    list.append('Rs_')
                    num += 1
            if model in ['PJAFFE', 'PJAFFE_ELLIPSE']:
                if not 'Ra' in kwargs_fixed:
                    list.append('Ra_')
                    num += 1
            if model in ['SPEMD_SMOOTH']:
                if not 's_scale' in kwargs_fixed:
                    list.append('s_scale')
                    num += 1
            if model in ['SIS', 'SIE', 'SPP', 'SPEP', 'SPEMD', 'SPEMD_SMOOTH', 'NFW', 'TNFW', 'NFW_ELLIPSE', 'SIS_TRUNCATED', 'SHAPELETS_POLAR',
                         'SHAPELETS_CART', 'DIPOLE', 'GAUSSIAN_KAPPA', 'SERSIC', 'SERSIC_ELLIPSE', 'HERNQUIST',
                         'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                if not 'center_x' in kwargs_fixed:
                    num += 1
                    list.append('center_x_lens')
                if not 'center_y' in kwargs_fixed:
                    num += 1
                    list.append('center_y_lens')
            if model in ['INTERPOL_SCALED']:
                if not 'scale_factor' in kwargs_fixed:
                    list.append('scale_factor')
                    num += 1
        return num, list