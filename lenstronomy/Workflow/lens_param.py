import numpy as np
import astrofunc.util as util


class LensParam(object):
    """
    class to handle the lens model parameter
    """
    def __init__(self, kwargs_options, kwargs_fixed):
        """

        :param kwargs_options:
        :param kwargs_fixed:
        """
        self.kwargs_options = kwargs_options
        self.kwargs_fixed = kwargs_fixed
        self.num_images = kwargs_options.get('num_images', 4)
        if kwargs_options.get('solver', False):
            self.solver_type = kwargs_options.get('solver_type', 'SPEP')
        else:
            self.solver_type = None

    def getParams(self, args, i):
        kwargs = {}
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                kwargs['amp'] = args[i]
                i += 1
            if not 'sigma_x' in self.kwargs_fixed:
                kwargs['sigma_x'] = np.exp(args[i])
                i += 1
            if not 'sigma_y' in self.kwargs_fixed:
                kwargs['sigma_y'] = np.exp(args[i])
                i += 1
        elif self.kwargs_options['lens_type'] in ['SPEP', 'ELLIPSE', 'SPEP_SHAPELETS', 'SPEP_NFW', 'SPEP_SIS', 'SPEP_SPP', 'SPEP_SPP_SHAPELETS', 'SPEP_SPP_DIPOLE', 'SPEP_SPP_DIPOLE_SHAPELETS']:
        #elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed:
                kwargs['theta_E'] = args[i]
                i += 1
            if not 'gamma' in self.kwargs_fixed:
                kwargs['gamma'] = args[i]
                i += 1
            if not 'q' in self.kwargs_fixed or not 'phi_G' in self.kwargs_fixed:
                phi, q = util.elliptisity2phi_q(args[i], args[i+1])
                kwargs['phi_G'] = phi
                kwargs['q'] = q
                i += 2

        if self.kwargs_options['lens_type'] in ['NFW', 'SPEP_NFW']:
            if not 'Rs' in self.kwargs_fixed:
                kwargs['Rs'] = np.exp(args[i])
                i += 1
            if not 'rho0' in self.kwargs_fixed:
                kwargs['rho0'] = np.exp(args[i])
                i += 1
            if not 'r200' in self.kwargs_fixed:
                kwargs['r200'] = np.exp(args[i])
                i += 1
            if not 'center_x_nfw' in self.kwargs_fixed:
                kwargs['center_x_nfw'] = args[i]
                i += 1
            if not 'center_y_nfw' in self.kwargs_fixed:
                kwargs['center_y_nfw'] = args[i]
                i += 1

        if self.kwargs_options['lens_type'] in ['SIS', 'SPEP_SIS']:
            if not 'theta_E_sis' in self.kwargs_fixed:
                kwargs['theta_E_sis'] = np.exp(args[i])
                i += 1
            if not 'center_x_sis' in self.kwargs_fixed:
                kwargs['center_x_sis'] = args[i]
                i += 1
            if not 'center_y_sis' in self.kwargs_fixed:
                kwargs['center_y_sis'] = args[i]
                i += 1
        if self.kwargs_options['lens_type'] in ['SPP', 'SPEP_SPP', 'SPEP_SPP_SHAPELETS', 'SPEP_SPP_DIPOLE', 'SPEP_SPP_DIPOLE_SHAPELETS']:
            if not 'theta_E_spp' in self.kwargs_fixed:
                kwargs['theta_E_spp'] = np.exp(args[i])
                i += 1
            if not 'gamma_spp' in self.kwargs_fixed:
                kwargs['gamma_spp'] = args[i]
                i += 1
            if not 'center_x_spp' in self.kwargs_fixed:
                kwargs['center_x_spp'] = args[i]
                i+=1
            if not 'center_y_spp' in self.kwargs_fixed:
                kwargs['center_y_spp'] = args[i]
                i += 1
        if self.kwargs_options['lens_type'] in ['SHAPELETS_POLAR', 'SPEP_SHAPELETS', 'SPEP_SPP_SHAPELETS', 'SPEP_SPP_DIPOLE_SHAPELETS']:
            if not 'beta' in self.kwargs_fixed:
                kwargs['beta'] = args[i]
                i += 1
            if not 'coeffs' in self.kwargs_fixed:
                num_coeffs = self.kwargs_options['num_shapelet_lens']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        num_coeffs -= 6
                        coeffs = args[i:i+num_coeffs]
                        coeffs = [0,0,0,0,0,0] + list(coeffs[0:])
                    elif self.num_images == 2:
                        num_coeffs -=3
                        coeffs = args[i:i+num_coeffs]
                        coeffs = [0, 0, 0] + list(coeffs[0:])
                    kwargs['coeffs'] = coeffs
                else:
                    kwargs['coeffs'] = args[i:i+num_coeffs]
                i += num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed:
                kwargs['center_x_shape'] = args[i]
                i += 1
            if not 'center_y_shape' in self.kwargs_fixed:
                kwargs['center_y_shape'] = args[i]
                i += 1
        if self.kwargs_options['lens_type'] in ['SPEP_SPP_DIPOLE', 'SPEP_SPP_DIPOLE_SHAPELETS']:
            if not 'coupling' in self.kwargs_fixed:
                kwargs['coupling'] = args[i]
                i += 1
            if not 'phi_dipole' in self.kwargs_fixed and self.kwargs_options['phi_dipole_decoupling'] is True:
                kwargs['phi_dipole'] = args[i]
                i += 1
        if not self.kwargs_options['lens_type'] in ['INTERPOL', 'SIS', 'NFW', 'SPP', 'NONE']:
            if not 'center_x' in self.kwargs_fixed:
                kwargs['center_x'] = args[i]
                i += 1
            if not 'center_y' in self.kwargs_fixed:
                kwargs['center_y'] = args[i]
                i += 1
        return kwargs, i

    def setParams(self, kwargs):
        """

        :param kwargs:
        :return:
        """
        args = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                args.append(kwargs['amp'])
            if not 'sigma_x' in self.kwargs_fixed:
                args.append(np.log(kwargs['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed:
                args.append(np.log(kwargs['sigma_y']))

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed:
                args.append(kwargs['theta_E'])
            if not 'gamma' in self.kwargs_fixed:
                args.append(kwargs['gamma'])
            if not 'q' in self.kwargs_fixed or not 'phi_G' in self.kwargs_fixed:
                e1, e2 = util.phi_q2_elliptisity(kwargs['phi_G'], kwargs['q'])
                args.append(e1)
                args.append(e2)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed:
                args.append(np.log(kwargs['Rs']))
            if not 'rho0' in self.kwargs_fixed:
                args.append(np.log(kwargs['rho0']))
            if not 'r200' in self.kwargs_fixed:
                args.append(np.log(kwargs['r200']))
            if not 'center_x_nfw' in self.kwargs_fixed:
                args.append(kwargs['center_x_nfw'])
            if not 'center_y_nfw' in self.kwargs_fixed:
                args.append(kwargs['center_y_nfw'])

        if self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed:
                args.append(np.log(kwargs['theta_E_sis']))
            if not 'center_x_sis' in self.kwargs_fixed:
                args.append(kwargs['center_x_sis'])
            if not 'center_y_sis' in self.kwargs_fixed:
                args.append(kwargs['center_y_sis'])
        if self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed:
                args.append(np.log(kwargs['theta_E_spp']))
            if not 'gamma_spp' in self.kwargs_fixed:
                args.append(kwargs['gamma_spp'])
            if not 'center_x_spp' in self.kwargs_fixed:
                args.append(kwargs['center_x_spp'])
            if not 'center_y_spp' in self.kwargs_fixed:
                args.append(kwargs['center_y_spp'])

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed:
                args.append(kwargs['beta'])
            if not 'coeffs' in self.kwargs_fixed:
                coeffs = kwargs['coeffs']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        coeffs = coeffs[6:]
                    elif self.num_images == 2:
                        coeffs = coeffs[3:]
                args += list(coeffs)
            if not 'center_x_shape' in self.kwargs_fixed:
                args.append(kwargs['center_x_shape'])
            if not 'center_y_shape' in self.kwargs_fixed:
                args.append(kwargs['center_y_shape'])

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed:
                args.append(kwargs['coupling'])
            if not 'phi_dipole' in self.kwargs_fixed and self.kwargs_options['phi_dipole_decoupling'] is True:
                args.append(kwargs['phi_dipole'])

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed:
                args.append(kwargs['center_x'])
            if not 'center_y' in self.kwargs_fixed:
                args.append(kwargs['center_y'])
        return args

    def add2fix(self, kwargs_fixed):
        """

        :param kwargs_fixed:
        :return:
        """
        fix_return = {}
        kwargs_fixed = kwargs_fixed
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if 'amp' in kwargs_fixed:
                fix_return['amp'] = kwargs_fixed['amp']
            if 'sigma_x' in kwargs_fixed:
                fix_return['sigma_x'] = kwargs_fixed['sigma_x']
            if 'sigma_y' in kwargs_fixed:
                fix_return['sigma_y'] = kwargs_fixed['sigma_y']

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'theta_E' in kwargs_fixed:
                fix_return['theta_E'] = kwargs_fixed['theta_E']
            if 'gamma' in kwargs_fixed:
                fix_return['gamma'] = kwargs_fixed['gamma']
            if 'q' in kwargs_fixed and 'phi_G' in kwargs_fixed:
                fix_return['phi_G'] = kwargs_fixed['phi_G']
                fix_return['q'] = kwargs_fixed['q']

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if 'Rs' in kwargs_fixed:
                fix_return['Rs'] = kwargs_fixed['Rs']
            if 'rho0' in kwargs_fixed:
                fix_return['rho0'] = kwargs_fixed['rho0']
            if 'r200' in kwargs_fixed:
                fix_return['r200'] = kwargs_fixed['r200']
            if 'center_x_nfw' in kwargs_fixed:
                fix_return['center_x_nfw'] = kwargs_fixed['center_x_nfw']
            if 'center_y_nfw' in kwargs_fixed:
                fix_return['center_y_nfw'] = kwargs_fixed['center_y_nfw']

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if 'theta_E_sis' in kwargs_fixed:
                fix_return['theta_E_sis'] = kwargs_fixed['theta_E_sis']
            if 'center_x_sis' in kwargs_fixed:
                fix_return['center_x_sis'] = kwargs_fixed['center_x_sis']
            if 'center_y_sis' in kwargs_fixed:
                fix_return['center_y_sis'] = kwargs_fixed['center_y_sis']

        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'theta_E_spp' in kwargs_fixed:
                fix_return['theta_E_spp'] = kwargs_fixed['theta_E_spp']
            if 'gamma_spp' in kwargs_fixed:
                fix_return['gamma_spp'] = kwargs_fixed['gamma_spp']
            if 'center_x_spp' in kwargs_fixed:
                fix_return['center_x_spp'] = kwargs_fixed['center_x_spp']
            if 'center_y_spp' in kwargs_fixed:
                fix_return['center_y_spp'] = kwargs_fixed['center_y_spp']

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'beta' in kwargs_fixed:
                fix_return['beta'] = kwargs_fixed['beta']
            if 'coeffs' in kwargs_fixed:
                fix_return['coeffs'] = kwargs_fixed['coeffs']
            if 'center_x_shape' in kwargs_fixed:
                fix_return['center_x_shape'] = kwargs_fixed['center_x_shape']
            if 'center_y_shape' in kwargs_fixed:
                fix_return['center_y_shape'] = kwargs_fixed['center_y_shape']

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'coupling' in kwargs_fixed:
                fix_return['coupling'] = kwargs_fixed['coupling']
            if 'phi_dipole' in kwargs_fixed:
                fix_return['phi_dipole'] = kwargs_fixed['phi_dipole']

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE'):
            if 'center_x' in kwargs_fixed:
                fix_return['center_x'] = kwargs_fixed['center_x']
            if 'center_y' in kwargs_fixed:
                fix_return['center_y'] = kwargs_fixed['center_y']
        return fix_return

    def param_init(self, kwargs_mean):
        """

        :param kwargs_mean:
        :return:
        """
        mean = []
        sigma = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                mean.append(kwargs_mean['amp'])
                sigma.append(kwargs_mean['amp_sigma'])
            if not 'sigma_x' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['sigma_x']))
                sigma.append(np.log(1 + kwargs_mean['sigma_x_sigma']/kwargs_mean['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['sigma_y']))
                sigma.append(np.log(1 + kwargs_mean['sigma_y_sigma']/kwargs_mean['sigma_y']))

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed:
                mean.append(kwargs_mean['theta_E'])
                sigma.append(kwargs_mean['theta_E_sigma'])
            if not 'gamma' in self.kwargs_fixed:
                mean.append(kwargs_mean['gamma'])
                sigma.append(kwargs_mean['gamma_sigma'])
            if not 'q' in self.kwargs_fixed or not 'phi_G' in self.kwargs_fixed:
                phi = kwargs_mean['phi_G']
                q = kwargs_mean['q']
                e1, e2 = util.phi_q2_elliptisity(phi, q)
                mean.append(e1)
                mean.append(e2)
                ellipse_sigma = kwargs_mean['ellipse_sigma']
                sigma.append(ellipse_sigma)
                sigma.append(ellipse_sigma)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['Rs']))
                sigma.append(np.log(1 + kwargs_mean['Rs_sigma']/kwargs_mean['Rs']))
            if not 'rho0' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['rho0']))
                sigma.append(np.log(1 + kwargs_mean['rho0_sigma']/kwargs_mean['rho0']))
            if not 'r200' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['r200']))
                sigma.append(np.log(1 + kwargs_mean['r200_sigma']/kwargs_mean['r200']))
            if not 'center_x_nfw' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x_nfw'])
                sigma.append(kwargs_mean['center_x_nfw_sigma'])
            if not 'center_y_nfw' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y_nfw'])
                sigma.append(kwargs_mean['center_y_nfw_sigma'])

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['theta_E_sis']))
                sigma.append(kwargs_mean['theta_E_sis_sigma'])
            if not 'center_x_sis' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x_sis'])
                sigma.append(kwargs_mean['center_x_sis_sigma'])
            if not 'center_y_sis' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y_sis'])
                sigma.append(kwargs_mean['center_y_sis_sigma'])

        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed:
                mean.append(np.log(kwargs_mean['theta_E_spp']))
                sigma.append(kwargs_mean['theta_E_spp_sigma'])
            if not 'gamma_spp' in self.kwargs_fixed:
                mean.append(kwargs_mean['gamma_spp'])
                sigma.append(kwargs_mean['gamma_spp_sigma'])
            if not 'center_x_spp' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x_spp'])
                sigma.append(kwargs_mean['center_x_spp_sigma'])
            if not 'center_y_spp' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y_spp'])
                sigma.append(kwargs_mean['center_y_spp_sigma'])

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed:
                mean.append(kwargs_mean['beta'])
                sigma.append(kwargs_mean['beta_sigma'])
            if not 'coeffs' in self.kwargs_fixed:
                coeffs = kwargs_mean['coeffs']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        coeffs = coeffs[6:]
                    elif self.num_images == 2:
                        coeffs = coeffs[3:]
                for i in range(0, len(coeffs)):
                    mean.append(coeffs[i])
                    sigma.append(kwargs_mean['coeffs_sigma'])
            if not 'center_x_shape' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x_shape'])
                sigma.append(kwargs_mean['center_x_shape_sigma'])
            if not 'center_y_shape' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y_shape'])
                sigma.append(kwargs_mean['center_y_shape_sigma'])

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed:
                mean.append(kwargs_mean['coupling'])
                sigma.append(kwargs_mean['coupling_sigma'])
            if not 'phi_dipole' in self.kwargs_fixed and self.kwargs_options['phi_dipole_decoupling'] is True:
                mean.append(kwargs_mean['phi_dipole'])
                sigma.append(kwargs_mean['phi_dipole_sigma'])

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_x'])
                sigma.append(kwargs_mean['center_x_sigma'])
            if not 'center_y' in self.kwargs_fixed:
                mean.append(kwargs_mean['center_y'])
                sigma.append(kwargs_mean['center_y_sigma'])
        return mean, sigma

    def param_bounds(self):
        """

        :return:
        """
        low = []
        high = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                low.append(0)
                high.append(1000)
            if not 'sigma_x' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)
            if not 'sigma_y' in self.kwargs_fixed:
                low.append(-10)
                high.append(10)

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS'  or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed:
                low.append(0.001)
                high.append(10)
            if not 'gamma' in self.kwargs_fixed:
                low.append(1.)
                high.append(2.85)
            if not 'q' in self.kwargs_fixed or not 'phi_G' in self.kwargs_fixed:
                low.append(-0.8)
                high.append(0.8)
                low.append(-0.8)
                high.append(0.8)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed:
                low.append(-5)
                high.append(5)
            if not 'rho0' in self.kwargs_fixed:
                low.append(-5)
                high.append(5)
            if not 'r200' in self.kwargs_fixed:
                low.append(-5)
                high.append(5)
            if not 'center_x_nfw' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
            if not 'center_y_nfw' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed:
                low.append(-10)
                high.append(1)
            if not 'center_x_sis' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
            if not 'center_y_sis' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed:
                low.append(-10)
                high.append(5)
            if not 'gamma_spp' in self.kwargs_fixed:
                low.append(1.45)
                high.append(2.85)
            if not 'center_x_spp' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)
            if not 'center_y_spp' in self.kwargs_fixed:
                low.append(-3)
                high.append(3)

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed:
                low.append(0.1)
                high.append(3.)
            if not 'coeffs' in self.kwargs_fixed:
                num_coeffs = self.kwargs_options['num_shapelet_lens']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        num_coeffs -= 6
                    elif self.num_images == 2:
                        num_coeffs -= 3
                low += [-5]*num_coeffs
                high += [5]*num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed:
                low.append(-2)
                high.append(2)
            if not 'center_y_shape' in self.kwargs_fixed:
                low.append(-2)
                high.append(2)

        if self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed:
                low.append(0)
                high.append(10)
            if not 'phi_dipole' in self.kwargs_fixed and self.kwargs_options['phi_dipole_decoupling'] is True:
                low.append(-np.pi)
                high.append(+np.pi)

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed:
                low.append(-20)
                high.append(20)
            if not 'center_y' in self.kwargs_fixed:
                low.append(-20)
                high.append(20)
        return low, high

    def num_param(self):
        """

        :return:
        """
        num = 0
        list = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed:
                num+=1
                list.append('amp_lens')
            if not 'sigma_x' in self.kwargs_fixed:
                num+=1
                list.append('sigma_x_lens')
            if not 'sigma_y' in self.kwargs_fixed:
                num+=1
                list.append('sigma_y_lens')
        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed:
                num+=1
                list.append('theta_E_lens')
            if not 'gamma' in self.kwargs_fixed:
                num+=1
                list.append('gamma_lens')
            if not 'q' in self.kwargs_fixed or not 'phi_G' in self.kwargs_fixed:
                num+=2
                list.append('e1_lens')
                list.append('e2_lens')
        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed:
                num+=1
                list.append('Rs_nfw')
            if not 'rho0' in self.kwargs_fixed:
                num+=1
                list.append('rho0_nfw')
            if not 'r200' in self.kwargs_fixed:
                num+=1
                list.append('r200_nfw')
            if not 'center_x_nfw' in self.kwargs_fixed:
                num+=1
                list.append('center_x_nfw')
            if not 'center_y_nfw' in self.kwargs_fixed:
                num+=1
                list.append('center_y_nfw')
        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed:
                num+=1
                list.append('theta_E_sis')
            if not 'center_x_sis' in self.kwargs_fixed:
                num+=1
                list.append('center_x_sis')
            if not 'center_y_sis' in self.kwargs_fixed:
                num+=1
                list.append('center_y_sis')
        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed:
                num+=1
                list.append('theta_E_spp')
            if not 'gamma_spp' in self.kwargs_fixed:
                num+=1
                list.append('gamma_spp')
            if not 'center_x_spp' in self.kwargs_fixed:
                num+=1
                list.append('center_x_spp')
            if not 'center_y_spp' in self.kwargs_fixed:
                num+=1
                list.append('center_y_spp')

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed:
                num+=1
                list.append('beta_lens')
            if not 'coeffs' in self.kwargs_fixed:
                num_coeffs = self.kwargs_options['num_shapelet_lens']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        num_coeffs -= 6
                    elif self.num_images == 2:
                        num_coeffs -= 3
                num += num_coeffs
                list += ['coeff']*num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed:
                num += 1
                list.append('center_x_lens_shape')
            if not 'center_y_shape' in self.kwargs_fixed:
                num += 1
                list.append('center_y_lens_shape')

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed:
                num += 1
                list.append('coupling')
            if not 'phi_dipole' in self.kwargs_fixed and self.kwargs_options['phi_dipole_decoupling'] is True:
                num += 1
                list.append('phi_dipole')

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed:
                num+=1
                list.append('center_x_lens')
            if not 'center_y' in self.kwargs_fixed:
                num+=1
                list.append('center_y_lens')
        return num, list