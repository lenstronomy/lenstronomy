__author__ = 'sibirrer'


import numpy as np
import astrofunc.util as util

from lenstronomy.MCMC.solver4point import Constraints
from lenstronomy.MCMC.solver2point import Constraints2
from lenstronomy.ImSim.make_image import MakeImage

class Param(object):
    """
    this class contains routines to deal with the number of parameters given certain options in a config file

    rule: first come the lens parameters, than the source parameters, psf parameters and at the end (if needed) some more

    list of parameters
    Gaussian: amp, sigma_x, sigma_y (center_x, center_y as options)
    NFW: to do
    SIS:  phi_E, (center_x, center_y as options)
    SPEMD: to do
    SPEP:  phi_E,gamma,q,phi_G, (center_x, center_y as options)
    """

    def __init__(self, kwargs_options, kwargs_fixed_lens={}, kwargs_fixed_source={}, kwargs_fixed_lens_light={}, kwargs_fixed_else={}):
        """

        :return:
        """
        self.kwargs_fixed_lens = kwargs_fixed_lens
        self.kwargs_fixed_source = kwargs_fixed_source
        self.kwargs_fixed_lens_light = kwargs_fixed_lens_light
        self.kwargs_fixed_else = kwargs_fixed_else
        self.kwargs_options = kwargs_options
        self.makeImage = MakeImage(kwargs_options)
        if kwargs_options['lens_type'] == 'SPEP_SIS':
            self.clump_type = 'SIS'
        elif kwargs_options['lens_type'] == 'SPEP_NFW':
            self.clump_type = 'NFW'
        elif kwargs_options['lens_type'] == 'SPEP_SPP':
            self.clump_type = 'SPP'
        elif kwargs_options['lens_type'] == 'SPEP_SHAPELETS':
            self.clump_type = 'SHAPELETS'
        elif kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS':
            self.clump_type = 'SPP_SHAPELETS'
        elif kwargs_options['lens_type'] == 'SPEP':
            self.clump_type = 'NO_clump'
        else:
            self.clump_type = None
        self.external_shear = kwargs_options.get('external_shear', False)
        self.foreground_shear = kwargs_options.get('foreground_shear', False) \
                                and kwargs_options.get('external_shear', False)
        self.num_images = kwargs_options.get('num_images', 4)
        if kwargs_options.get('solver', False):
            self.solver_type = kwargs_options.get('solver_type', 'SPEP')
            if self.num_images == 4:
                self.constraints = Constraints(self.solver_type)
            elif self. num_images == 2:
                self.constraints = Constraints2(self.solver_type)
            else:
                raise ValueError("%s number of images is not valid. Use 2 or 4!" % self.num_images)
        else:
            self.solver_type = None

    def getParams(self, args):
        """

        :param args: tuple of parameter values (float, strings, ...(
        :return: keyword arguments sorted
        """
        i = 0
        kwargs_lens = {}
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_lens:
                kwargs_lens['amp'] = args[i]
                i += 1
            if not 'sigma_x' in self.kwargs_fixed_lens:
                kwargs_lens['sigma_x'] = np.exp(args[i])
                i += 1
            if not 'sigma_y' in self.kwargs_fixed_lens:
                kwargs_lens['sigma_y'] = np.exp(args[i])
                i += 1

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed_lens:
                kwargs_lens['theta_E'] = args[i]
                i += 1
            if not 'gamma' in self.kwargs_fixed_lens:
                kwargs_lens['gamma'] = args[i]
                i += 1
            if not 'q' in self.kwargs_fixed_lens or not 'phi_G' in self.kwargs_fixed_lens:
                phi, q = util.elliptisity2phi_q(args[i], args[i+1])
                kwargs_lens['phi_G'] = phi
                kwargs_lens['q'] = q
                i += 2

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed_lens:
                kwargs_lens['Rs'] = np.exp(args[i])
                i += 1
            if not 'rho0' in self.kwargs_fixed_lens:
                kwargs_lens['rho0'] = np.exp(args[i])
                i += 1
            if not 'r200' in self.kwargs_fixed_lens:
                kwargs_lens['r200'] = np.exp(args[i])
                i += 1
            if not 'center_x_nfw' in self.kwargs_fixed_lens:
                kwargs_lens['center_x_nfw'] = args[i]
                i += 1
            if not 'center_y_nfw' in self.kwargs_fixed_lens:
                kwargs_lens['center_y_nfw'] = args[i]
                i += 1

        if self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed_lens:
                kwargs_lens['theta_E_sis'] = np.exp(args[i])
                i += 1
            if not 'center_x_sis' in self.kwargs_fixed_lens:
                kwargs_lens['center_x_sis'] = args[i]
                i += 1
            if not 'center_y_sis' in self.kwargs_fixed_lens:
                kwargs_lens['center_y_sis'] = args[i]
                i += 1
        if self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed_lens:
                kwargs_lens['theta_E_spp'] = np.exp(args[i])
                i += 1
            if not 'gamma_spp' in self.kwargs_fixed_lens:
                kwargs_lens['gamma_spp'] = args[i]
                i += 1
            if not 'center_x_spp' in self.kwargs_fixed_lens:
                kwargs_lens['center_x_spp'] = args[i]
                i+=1
            if not 'center_y_spp' in self.kwargs_fixed_lens:
                kwargs_lens['center_y_spp'] = args[i]
                i += 1

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed_lens:
                kwargs_lens['beta'] = args[i]
                i += 1
            if not 'coeffs' in self.kwargs_fixed_lens:
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
                    kwargs_lens['coeffs'] = coeffs
                else:
                    kwargs_lens['coeffs'] = args[i:i+num_coeffs]
                i += num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed_lens:
                kwargs_lens['center_x_shape'] = args[i]
                i += 1
            if not 'center_y_shape' in self.kwargs_fixed_lens:
                kwargs_lens['center_y_shape'] = args[i]
                i += 1

        if self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed_lens:
                kwargs_lens['coupling'] = args[i]
                i += 1
            if not 'phi_dipole' in self.kwargs_fixed_else and self.kwargs_options['phi_dipole_decoupling'] is True:
                kwargs_lens['phi_dipole'] = args[i]
                i += 1

        if not (self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed_lens:
                kwargs_lens['center_x'] = args[i]
                i += 1
            if not 'center_y' in self.kwargs_fixed_lens:
                kwargs_lens['center_y'] = args[i]
                i += 1

        kwargs_source = {}
        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_source:
                kwargs_source['amp'] = args[i]
                i += 1
            if not 'sigma_x' in self.kwargs_fixed_source:
                kwargs_source['sigma_x'] = np.exp(args[i])
                i += 1
            if not 'sigma_y' in self.kwargs_fixed_source:
                kwargs_source['sigma_y'] = np.exp(args[i])
                i += 1

        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed_source:
                kwargs_source['I0_sersic'] = args[i]
                i += 1
            if not 'n_sersic' in self.kwargs_fixed_source:
                kwargs_source['n_sersic'] = args[i]
                i += 1
            if not 'R_sersic' in self.kwargs_fixed_source:
                kwargs_source['R_sersic'] = args[i]
                i += 1
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed_source or not 'q' in self.kwargs_fixed_source:
                    phi, q = util.elliptisity2phi_q(args[i],args[i+1])
                    kwargs_source['phi_G'] = phi
                    kwargs_source['q'] = q
                    i += 2
        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed_source:
                kwargs_source['center_x'] = args[i]
                i += 1
            if not 'center_y' in self.kwargs_fixed_source:
                kwargs_source['center_y'] = args[i]
                i += 1

        kwargs_lens_light = {}
        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_sersic' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['I0_sersic'] = args[i]
                i += 1
            if not 'n_sersic' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['n_sersic'] = args[i]
                i += 1
            if not 'R_sersic' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['R_sersic'] = args[i]
                i += 1
            if not 'center_x' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['center_x'] = args[i]
                i += 1
            if not 'center_y' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['center_y'] = args[i]
                i += 1
        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if not 'phi_G' in self.kwargs_fixed_lens_light or not 'q' in self.kwargs_fixed_lens_light:
                    phi, q = util.elliptisity2phi_q(args[i], args[i+1])
                    kwargs_lens_light['phi_G'] = phi
                    kwargs_lens_light['q'] = q
                    i += 2

        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_2' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['I0_2'] = args[i]
                i += 1
            if not 'R_2' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['R_2'] = args[i]
                i += 1
            if not 'n_2' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['n_2'] = args[i]
                i += 1
            if not 'center_x_2' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['center_x_2'] = args[i]
                i += 1
            if not 'center_y_2' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['center_y_2'] = args[i]
                i += 1
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if not 'Rb' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['Rb'] = args[i]
                i += 1
            if not 'gamma' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['gamma'] = args[i]
                i += 1
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['I0_3'] = args[i]
                i += 1
            if not 'R_3' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['R_3'] = args[i]
                i += 1
            if not 'n_3' in self.kwargs_fixed_lens_light:
                kwargs_lens_light['n_3'] = args[i]
                i += 1

        kwargs_else = {}
        if not 'ra_pos' in self.kwargs_fixed_else:
            kwargs_else['ra_pos'] = np.array(args[i:i+self.num_images])
            i += self.num_images
        if not 'dec_pos' in self.kwargs_fixed_else:
            kwargs_else['dec_pos'] = np.array(args[i:i+self.num_images])
            i += self.num_images
        if self.kwargs_options.get('image_plane_source', False):
            if not 'source_pos_image_ra' in self.kwargs_fixed_else:
                kwargs_else['source_pos_image_ra'] = args[i]
                i += 1
            if not 'source_pos_image_dec' in self.kwargs_fixed_else:
                kwargs_else['source_pos_image_dec'] = args[i]
                i += 1
        if self.external_shear:
            if not 'gamma1' in self.kwargs_fixed_else or not 'gamma2' in self.kwargs_fixed_else:
                kwargs_else['gamma1'] = args[i]
                kwargs_else['gamma2'] = args[i+1]
                i += 2
        if self.foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed_else or not 'gamma2_foreground' in self.kwargs_fixed_else:
                kwargs_else['gamma1_foreground'] = args[i]
                kwargs_else['gamma2_foreground'] = args[i+1]
                i += 2
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed_else:
                kwargs_else['delay_dist'] = args[i]
                i += 1
        if not 'shapelet_beta' in self.kwargs_fixed_else:
            kwargs_else['shapelet_beta'] = args[i]
            i += 1
        if self.kwargs_options.get('add_clump', False):
            if not 'theta_E_clump' in self.kwargs_fixed_else:
                kwargs_else['theta_E_clump'] = args[i]
                i += 1
            if not 'r_trunc' in self.kwargs_fixed_else:
                kwargs_else['r_trunc'] = args[i]
                i += 1
            if not 'x_clump' in self.kwargs_fixed_else:
                kwargs_else['x_clump'] = args[i]
                i += 1
            if not 'y_clump' in self.kwargs_fixed_else:
                kwargs_else['y_clump'] = args[i]
                i += 1
        if self.kwargs_options.get('psf_iteration', False):
            if not 'point_amp' in self.kwargs_fixed_else:
                point_amp = np.append(1, args[i:i+3])
                kwargs_else['point_amp'] = point_amp
                i += 3
        lens_dict = dict(kwargs_lens.items() + self.kwargs_fixed_lens.items())
        source_dict = dict(kwargs_source.items() + self.kwargs_fixed_source.items())
        lens_light_dict = dict(kwargs_lens_light.items() + self.kwargs_fixed_lens_light.items())
        else_dict = dict(kwargs_else.items() + self.kwargs_fixed_else.items())
        return lens_dict, source_dict, lens_light_dict, else_dict

    def setParams(self, kwargs_lens, kwargs_source, kwargs_lens_light={}, kwargs_else={}):
        """
        inverse of getParam function
        :param kwargs_lens: keyword arguments depending on model options
        :param kwargs_source: keyword arguments depending on model options
        :return: tuple of parameters
        """
        args = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['amp'])
            if not 'sigma_x' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['sigma_y']))

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['theta_E'])
            if not 'gamma' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['gamma'])
            if not 'q' in self.kwargs_fixed_lens or not 'phi_G' in self.kwargs_fixed_lens:
                e1, e2 = util.phi_q2_elliptisity(kwargs_lens['phi_G'], kwargs_lens['q'])
                args.append(e1)
                args.append(e2)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['Rs']))
            if not 'rho0' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['rho0']))
            if not 'r200' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['r200']))
            if not 'center_x_nfw' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_x_nfw'])
            if not 'center_y_nfw' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_y_nfw'])

        if self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['theta_E_sis']))
            if not 'center_x_sis' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_x_sis'])
            if not 'center_y_sis' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_y_sis'])
        if self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed_lens:
                args.append(np.log(kwargs_lens['theta_E_spp']))
            if not 'gamma_spp' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['gamma_spp'])
            if not 'center_x_spp' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_x_spp'])
            if not 'center_y_spp' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_y_spp'])

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['beta'])
            if not 'coeffs' in self.kwargs_fixed_lens:
                coeffs = kwargs_lens['coeffs']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        coeffs = coeffs[6:]
                    elif self.num_images == 2:
                        coeffs = coeffs[3:]
                args += list(coeffs)
            if not 'center_x_shape' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_x_shape'])
            if not 'center_y_shape' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_y_shape'])

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['coupling'])
            if not 'phi_dipole' in self.kwargs_fixed_else and self.kwargs_options['phi_dipole_decoupling'] is True:
                args.append(kwargs_lens['phi_dipole'])

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_x'])
            if not 'center_y' in self.kwargs_fixed_lens:
                args.append(kwargs_lens['center_y'])

        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_source:
                args.append(kwargs_source['amp'])
            if not 'sigma_x' in self.kwargs_fixed_source:
                args.append(np.log(kwargs_source['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed_source:
                args.append(np.log(kwargs_source['sigma_y']))

        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed_source:
                args.append(kwargs_source['I0_sersic'])
            if not 'n_sersic' in self.kwargs_fixed_source:
                args.append(kwargs_source['n_sersic'])
            if not 'R_sersic' in self.kwargs_fixed_source:
                args.append(kwargs_source['R_sersic'])
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed_source or not 'q' in self.kwargs_fixed_source:
                    e1, e2 = util.phi_q2_elliptisity(kwargs_lens['phi_G'], kwargs_lens['q'])
                    args.append(e1)
                    args.append(e2)
        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed_source:
                args.append(kwargs_source['center_x'])
            if not 'center_y' in self.kwargs_fixed_source:
                args.append(kwargs_source['center_y'])

        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_sersic' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['I0_sersic'])
            if not 'n_sersic' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['n_sersic'])
            if not 'R_sersic' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['R_sersic'])
            if not 'center_x' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['center_x'])
            if not 'center_y' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['center_y'])
        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if not 'phi_G' in self.kwargs_fixed_lens_light or not 'q' in self.kwargs_fixed_lens_light:
                    e1, e2 = util.phi_q2_elliptisity(kwargs_lens_light['phi_G'], kwargs_lens_light['q'])
                    args.append(e1)
                    args.append(e2)
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_2' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['I0_2'])
            if not 'R_2' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['R_2'])
            if not 'n_2' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['n_2'])
            if not 'center_x_2' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['center_x_2'])
            if not 'center_y_2' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['center_y_2'])
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if not 'Rb' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['Rb'])
            if not 'gamma' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['gamma'])
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['I0_3'])
            if not 'R_3' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['R_3'])
            if not 'n_3' in self.kwargs_fixed_lens_light:
                args.append(kwargs_lens_light['n_3'])

        if not 'ra_pos' in self.kwargs_fixed_else:
            x_pos = kwargs_else['ra_pos']
            for i in x_pos:
                args.append(i)
        if not 'dec_pos' in self.kwargs_fixed_else:
            y_pos = kwargs_else['dec_pos']
            for i in y_pos:
                args.append(i)
        if self.kwargs_options.get('image_plane_source', False):
            if not 'source_pos_image_ra' in self.kwargs_fixed_else:
                args.append(kwargs_else['source_pos_image_ra'])
            if not 'source_pos_image_dec' in self.kwargs_fixed_else:
                args.append(kwargs_else['source_pos_image_dec'])
        if self.external_shear:
            if not 'gamma1' in self.kwargs_fixed_else or not 'gamma2' in self.kwargs_fixed_else:
                args.append(kwargs_else['gamma1'])
                args.append(kwargs_else['gamma2'])
        if self.foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed_else or not 'gamma2_foreground' in self.kwargs_fixed_else:
                args.append(kwargs_else['gamma1_foreground'])
                args.append(kwargs_else['gamma2_foreground'])
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed_else:
                args.append(kwargs_else['delay_dist'])
        if not 'shapelet_beta' in self.kwargs_fixed_else:
            args.append(kwargs_else['shapelet_beta'])

        if self.kwargs_options.get('add_clump', False):
            if not 'theta_E_clump' in self.kwargs_fixed_else:
                args.append(kwargs_else['theta_E_clump'])
            if not 'r_trunc' in self.kwargs_fixed_else:
                args.append(kwargs_else['r_trunc'])
            if not 'x_clump' in self.kwargs_fixed_else:
                args.append(kwargs_else['x_clump'])
            if not 'y_clump' in self.kwargs_fixed_else:
                args.append(kwargs_else['y_clump'])
        if self.kwargs_options.get('psf_iteration', False):
            if not 'point_amp' in self.kwargs_fixed_else:
                point_amp = kwargs_else['point_amp']
                for i in point_amp[1:]:
                    args.append(i)
        return args

    def add_to_fixed(self, lens_fixed, source_fixed, lens_light_fixed, else_fixed):
        """
        changes the kwargs fixed with the inputs, if options are chosen such that it is modeled
        :param lens_fixed:
        :param source_fixed:
        :param lens_light_fixed:
        :param else_fixed:
        :return:
        """
        lens_fix = {}
        source_fix = {}
        lens_light_fix = {}
        else_fix = {}
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if 'amp' in lens_fixed:
                lens_fix['amp'] = lens_fixed['amp']
            if 'sigma_x' in lens_fixed:
                lens_fix['sigma_x'] = lens_fixed['sigma_x']
            if 'sigma_y' in lens_fixed:
                lens_fix['sigma_y'] = lens_fixed['sigma_y']

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'theta_E' in lens_fixed:
                lens_fix['theta_E'] = lens_fixed['theta_E']
            if 'gamma' in lens_fixed:
                lens_fix['gamma'] = lens_fixed['gamma']
            if 'q' in lens_fixed and 'phi_G' in lens_fixed:
                lens_fix['phi_G'] = lens_fixed['phi_G']
                lens_fix['q'] = lens_fixed['q']

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if 'Rs' in lens_fixed:
                lens_fix['Rs'] = lens_fixed['Rs']
            if 'rho0' in lens_fixed:
                lens_fix['rho0'] = lens_fixed['rho0']
            if 'r200' in lens_fixed:
                lens_fix['r200'] = lens_fixed['r200']
            if 'center_x_nfw' in lens_fixed:
                lens_fix['center_x_nfw'] = lens_fixed['center_x_nfw']
            if 'center_y_nfw' in lens_fixed:
                lens_fix['center_y_nfw'] = lens_fixed['center_y_nfw']

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if 'theta_E_sis' in lens_fixed:
                lens_fix['theta_E_sis'] = lens_fixed['theta_E_sis']
            if 'center_x_sis' in lens_fixed:
                lens_fix['center_x_sis'] = lens_fixed['center_x_sis']
            if 'center_y_sis' in lens_fixed:
                lens_fix['center_y_sis'] = lens_fixed['center_y_sis']

        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'theta_E_spp' in lens_fixed:
                lens_fix['theta_E_spp'] = lens_fixed['theta_E_spp']
            if 'gamma_spp' in lens_fixed:
                lens_fix['gamma_spp'] = lens_fixed['gamma_spp']
            if 'center_x_spp' in lens_fixed:
                lens_fix['center_x_spp'] = lens_fixed['center_x_spp']
            if 'center_y_spp' in lens_fixed:
                lens_fix['center_y_spp'] = lens_fixed['center_y_spp']

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'beta' in lens_fixed:
                lens_fix['beta'] = lens_fixed['beta']
            if 'coeffs' in lens_fixed:
                lens_fix['coeffs'] = lens_fixed['coeffs']
            if 'center_x_shape' in lens_fixed:
                lens_fix['center_x_shape'] = lens_fixed['center_x_shape']
            if 'center_y_shape' in lens_fixed:
                lens_fix['center_y_shape'] = lens_fixed['center_y_shape']

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if 'coupling' in lens_fixed:
                lens_fix['coupling'] = lens_fixed['coupling']
            if 'phi_dipole' in lens_fixed:
                lens_fix['phi_dipole'] = lens_fixed['phi_dipole']

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE'):
            if 'center_x' in lens_fixed:
                lens_fix['center_x'] = lens_fixed['center_x']
            if 'center_y' in lens_fixed:
                lens_fix['center_y'] = lens_fixed['center_y']

        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if 'amp' in source_fixed:
                source_fix['center_y'] = source_fixed['center_y']
            if 'sigma_x' in source_fixed:
                source_fix['sigma_x'] = source_fixed['sigma_x']
            if 'sigma_y' in source_fixed:
                source_fix['sigma_y'] = source_fixed['sigma_y']

        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if 'I0_sersic' in source_fixed:
                source_fix['I0_sersic'] = source_fixed['I0_sersic']
            if 'n_sersic' in source_fixed:
                source_fix['n_sersic'] = source_fixed['n_sersic']
            if 'R_sersic' in source_fixed:
                source_fix['R_sersic'] = source_fixed['R_sersic']
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if 'phi_G' in source_fixed and 'q' in source_fixed:
                    source_fix['phi_G'] = source_fixed['phi_G']
                    source_fix['q'] = source_fixed['q']
        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if 'center_x' in source_fixed:
                source_fix['center_x'] = source_fixed['center_x']
            if 'center_y' in source_fixed:
                source_fix['center_y'] = source_fixed['center_y']

        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if 'I0_sersic' in lens_light_fixed:
                lens_light_fix['I0_sersic'] = lens_light_fixed['I0_sersic']
            if 'n_sersic' in lens_light_fixed:
                lens_light_fix['n_sersic'] = lens_light_fixed['n_sersic']
            if 'R_sersic' in lens_light_fixed:
                lens_light_fix['R_sersic'] = lens_light_fixed['R_sersic']
            if 'center_x' in lens_light_fixed:
                lens_light_fix['center_x'] = lens_light_fixed['center_x']
            if 'center_y' in lens_light_fixed:
                lens_light_fix['center_y'] = lens_light_fixed['center_y']

        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if 'phi_G' in lens_light_fixed or 'q' in lens_light_fixed:
                    lens_light_fix['phi_G'] = lens_light_fixed['phi_G']
                    lens_light_fix['q'] = lens_light_fixed['q']

        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if 'I0_2' in lens_light_fixed:
                lens_light_fix['I0_2'] = lens_light_fixed['I0_2']
            if 'R_2' in lens_light_fixed:
                lens_light_fix['R_2'] = lens_light_fixed['R_2']
            if 'n_2' in lens_light_fixed:
                lens_light_fix['n_2'] = lens_light_fixed['n_2']
            if 'center_x_2' in lens_light_fixed:
                lens_light_fix['center_x_2'] = lens_light_fixed['center_x_2']
            if 'center_y_2' in lens_light_fixed:
                lens_light_fix['center_y_2'] = lens_light_fixed['center_y_2']
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if 'Rb' in lens_light_fixed:
                lens_light_fix['Rb'] = lens_light_fixed['Rb']
            if 'gamma' in lens_light_fixed:
                lens_light_fix['gamma'] = lens_light_fixed['gamma']
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if 'I0_3' in lens_light_fixed:
                lens_light_fix['I0_3'] = lens_light_fixed['I0_3']
            if 'R_3' in lens_light_fixed:
                lens_light_fix['R_3'] = lens_light_fixed['R_3']
            if 'n_3' in lens_light_fixed:
                lens_light_fix['n_3'] = lens_light_fixed['n_3']

        if 'ra_pos' in else_fixed:
            else_fix['ra_pos'] = else_fixed['ra_pos']
        if 'dec_pos' in else_fixed:
            else_fix['dec_pos'] = else_fixed['dec_pos']
        if self.kwargs_options.get('image_plane_source', False):
            if 'source_pos_image_ra' in else_fixed:
                else_fix['source_pos_image_ra'] = else_fixed['source_pos_image_ra']
            if 'source_pos_image_dec' in else_fixed:
                else_fix['source_pos_image_dec'] = else_fixed['source_pos_image_dec']
        if self.external_shear:
            if 'gamma1' in else_fixed:
                else_fix['gamma1'] = else_fixed['gamma1']
            if 'gamma2' in else_fixed:
                else_fix['gamma2'] = else_fixed['gamma2']

        if self.foreground_shear:
            if 'gamma1_foreground' in else_fixed:
                else_fix['gamma1_foreground'] = else_fixed['gamma1_foreground']
            if 'gamma2_foreground' in else_fixed:
                else_fix['gamma2_foreground'] = else_fixed['gamma2_foreground']

        if 'delay_dist' in else_fixed:
            else_fix['delay_dist'] = else_fixed['delay_dist']
        if self.kwargs_options.get('time_delay', False) is True:
            if 'delay_dist' in else_fixed:
                else_fix['delay_dist'] = else_fixed['delay_dist']
        if 'shapelet_beta' in else_fixed:
            else_fix['shapelet_beta'] = else_fixed['shapelet_beta']

        if self.kwargs_options.get('add_clump', False):
            if 'theta_E_clump' in else_fixed:
                else_fix['theta_E_clump'] = else_fixed['theta_E_clump']
            if 'r_trunc' in else_fixed:
                else_fix['r_trunc'] = else_fixed['r_trunc']
            if 'x_clump' in else_fixed:
                else_fix['x_clump'] = else_fixed['x_clump']
            if 'y_clump' in else_fixed:
                else_fix['y_clump'] = else_fixed['y_clump']
        if self.kwargs_options.get('psf_iteration', False):
            if 'point_amp' in else_fixed:
                else_fix['point_amp'] = else_fixed['point_amp']

        return lens_fix, source_fix, lens_light_fix, else_fix


    def param_init(self, kwarg_mean_lens, kwarg_mean_source, kwarg_mean_lens_light={}, kwarg_mean_else={}):
        """
        returns upper and lower bounds on the parameters used in the X2_chain function for MCMC/PSO starting
        bounds are defined relative to the catalogue level image called in the class Data
        might be migrated to the param class
        """
        #inizialize mean and sigma limit arrays
        mean = []
        sigma = []

        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['amp'])
                sigma.append(kwarg_mean_lens['amp_sigma'])
            if not 'sigma_x' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['sigma_x']))
                sigma.append(np.log(1 + kwarg_mean_lens['sigma_x_sigma']/kwarg_mean_lens['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['sigma_y']))
                sigma.append(np.log(1 + kwarg_mean_lens['sigma_y_sigma']/kwarg_mean_lens['sigma_y']))

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['theta_E'])
                sigma.append(kwarg_mean_lens['theta_E_sigma'])
            if not 'gamma' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['gamma'])
                sigma.append(kwarg_mean_lens['gamma_sigma'])
            if not 'q' in self.kwargs_fixed_lens or not 'phi_G' in self.kwargs_fixed_lens:
                phi = kwarg_mean_lens['phi_G']
                q = kwarg_mean_lens['q']
                e1, e2 = util.phi_q2_elliptisity(phi, q)
                mean.append(e1)
                mean.append(e2)
                ellipse_sigma = kwarg_mean_lens['ellipse_sigma']
                sigma.append(ellipse_sigma)
                sigma.append(ellipse_sigma)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['Rs']))
                sigma.append(np.log(1 + kwarg_mean_lens['Rs_sigma']/kwarg_mean_lens['Rs']))
            if not 'rho0' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['rho0']))
                sigma.append(np.log(1 + kwarg_mean_lens['rho0_sigma']/kwarg_mean_lens['rho0']))
            if not 'r200' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['r200']))
                sigma.append(np.log(1 + kwarg_mean_lens['r200_sigma']/kwarg_mean_lens['r200']))
            if not 'center_x_nfw' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_x_nfw'])
                sigma.append(kwarg_mean_lens['center_x_nfw_sigma'])
            if not 'center_y_nfw' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_y_nfw'])
                sigma.append(kwarg_mean_lens['center_y_nfw_sigma'])

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['theta_E_sis']))
                sigma.append(kwarg_mean_lens['theta_E_sis_sigma'])
            if not 'center_x_sis' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_x_sis'])
                sigma.append(kwarg_mean_lens['center_x_sis_sigma'])
            if not 'center_y_sis' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_y_sis'])
                sigma.append(kwarg_mean_lens['center_y_sis_sigma'])

        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed_lens:
                mean.append(np.log(kwarg_mean_lens['theta_E_spp']))
                sigma.append(kwarg_mean_lens['theta_E_spp_sigma'])
            if not 'gamma_spp' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['gamma_spp'])
                sigma.append(kwarg_mean_lens['gamma_spp_sigma'])
            if not 'center_x_spp' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_x_spp'])
                sigma.append(kwarg_mean_lens['center_x_spp_sigma'])
            if not 'center_y_spp' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_y_spp'])
                sigma.append(kwarg_mean_lens['center_y_spp_sigma'])

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['beta'])
                sigma.append(kwarg_mean_lens['beta_sigma'])
            if not 'coeffs' in self.kwargs_fixed_lens:
                coeffs = kwarg_mean_lens['coeffs']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        coeffs = coeffs[6:]
                    elif self.num_images == 2:
                        coeffs = coeffs[3:]
                for i in range(0, len(coeffs)):
                    mean.append(coeffs[i])
                    sigma.append(kwarg_mean_lens['coeffs_sigma'])
            if not 'center_x_shape' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_x_shape'])
                sigma.append(kwarg_mean_lens['center_x_shape_sigma'])
            if not 'center_y_shape' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_y_shape'])
                sigma.append(kwarg_mean_lens['center_y_shape_sigma'])

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['coupling'])
                sigma.append(kwarg_mean_lens['coupling_sigma'])
            if not 'phi_dipole' in self.kwargs_fixed_else and self.kwargs_options['phi_dipole_decoupling'] is True:
                mean.append(kwarg_mean_lens['phi_dipole'])
                sigma.append(kwarg_mean_lens['phi_dipole_sigma'])

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_x'])
                sigma.append(kwarg_mean_lens['center_x_sigma'])
            if not 'center_y' in self.kwargs_fixed_lens:
                mean.append(kwarg_mean_lens['center_y'])
                sigma.append(kwarg_mean_lens['center_y_sigma'])

        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['amp'])
                sigma.append(kwarg_mean_source['amp_sigma'])
            if not 'sigma_x' in self.kwargs_fixed_source:
                mean.append(np.log(kwarg_mean_source['sigma_x']))
                sigma.append(np.log(1 + kwarg_mean_source['sigma_x_sigma']/kwarg_mean_source['sigma_x']))
            if not 'sigma_y' in self.kwargs_fixed_source:
                mean.append(np.log(kwarg_mean_source['sigma_y']))
                sigma.append(np.log(1 + kwarg_mean_source['sigma_y_sigma']/kwarg_mean_source['sigma_y']))

        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['I0_sersic'])
                sigma.append(kwarg_mean_source['I0_sersic_sigma'])
            if not 'n_sersic' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['n_sersic'])
                sigma.append(kwarg_mean_source['n_sersic_sigma'])
            if not 'R_sersic' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['R_sersic'])
                sigma.append(kwarg_mean_source['R_sersic_sigma'])
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed_source or not 'q' in self.kwargs_fixed_source:
                    phi = kwarg_mean_source['phi_G']
                    q = kwarg_mean_source['q']
                    e1, e2 = util.phi_q2_elliptisity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwarg_mean_source['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)
        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['center_x'])
                sigma.append(kwarg_mean_source['center_x_sigma'])
            if not 'center_y' in self.kwargs_fixed_source:
                mean.append(kwarg_mean_source['center_y'])
                sigma.append(kwarg_mean_source['center_y_sigma'])

        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_sersic' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['I0_sersic'])
                sigma.append(kwarg_mean_lens_light['I0_sersic_sigma'])
            if not 'n_sersic' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['n_sersic'])
                sigma.append(kwarg_mean_lens_light['n_sersic_sigma'])
            if not 'R_sersic' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['R_sersic'])
                sigma.append(kwarg_mean_lens_light['R_sersic_sigma'])
            if not 'center_x' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['center_x'])
                sigma.append(kwarg_mean_lens_light['center_x_sigma'])
            if not 'center_y' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['center_y'])
                sigma.append(kwarg_mean_lens_light['center_y_sigma'])
        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if not 'phi_G' in self.kwargs_fixed_lens_light or not 'q' in self.kwargs_fixed_lens_light:
                    phi = kwarg_mean_lens_light['phi_G']
                    q = kwarg_mean_lens_light['q']
                    e1,e2 = util.phi_q2_elliptisity(phi, q)
                    mean.append(e1)
                    mean.append(e2)
                    ellipse_sigma = kwarg_mean_lens_light['ellipse_sigma']
                    sigma.append(ellipse_sigma)
                    sigma.append(ellipse_sigma)
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_2' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['I0_2'])
                sigma.append(kwarg_mean_lens_light['I0_2_sigma'])
            if not 'R_2' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['R_2'])
                sigma.append(kwarg_mean_lens_light['R_2_sigma'])
            if not 'n_2' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['n_2'])
                sigma.append(kwarg_mean_lens_light['n_2_sigma'])
            if not 'center_x_2' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['center_x_2'])
                sigma.append(kwarg_mean_lens_light['center_x_2_sigma'])
            if not 'center_y_2' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['center_y_2'])
                sigma.append(kwarg_mean_lens_light['center_y_2_sigma'])
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if not 'Rb' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['Rb'])
                sigma.append(kwarg_mean_lens_light['Rb_sigma'])
            if not 'gamma' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['gamma'])
                sigma.append(kwarg_mean_lens_light['gamma_sigma'])
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['I0_3'])
                sigma.append(kwarg_mean_lens_light['I0_3_sigma'])
            if not 'R_3' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['R_3'])
                sigma.append(kwarg_mean_lens_light['R_3_sigma'])
            if not 'n_3' in self.kwargs_fixed_lens_light:
                mean.append(kwarg_mean_lens_light['n_3'])
                sigma.append(kwarg_mean_lens_light['n_3_sigma'])

        if not 'ra_pos' in self.kwargs_fixed_else:
            x_pos_mean = kwarg_mean_else['ra_pos']
            pos_sigma = kwarg_mean_else['pos_sigma']
            for i in x_pos_mean:
                mean.append(i)
                sigma.append(pos_sigma)
        if not 'dec_pos' in self.kwargs_fixed_else:
            y_pos_mean = kwarg_mean_else['dec_pos']
            pos_sigma = kwarg_mean_else['pos_sigma']
            for i in y_pos_mean:
                mean.append(i)
                sigma.append(pos_sigma)
        if self.kwargs_options.get('image_plane_source', False):
            if not 'source_pos_image_ra' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['source_pos_image_ra'])
                sigma.append(kwarg_mean_else['source_pos_image_ra_sigma'])
            if not 'source_pos_image_dec' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['source_pos_image_dec'])
                sigma.append(kwarg_mean_else['source_pos_image_dec_sigma'])
        if self.external_shear:
            if not 'gamma1' in self.kwargs_fixed_else or not 'gamma2' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['gamma1'])
                mean.append(kwarg_mean_else['gamma2'])
                shear_sigma = kwarg_mean_else['shear_sigma']
                sigma.append(shear_sigma)
                sigma.append(shear_sigma)
        if self.foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed_else or not 'gamma2_foreground' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['gamma1_foreground'])
                mean.append(kwarg_mean_else['gamma2_foreground'])
                shear_sigma = kwarg_mean_else['shear_foreground_sigma']
                sigma.append(shear_sigma)
                sigma.append(shear_sigma)
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['delay_dist'])
                sigma.append(kwarg_mean_else['delay_dist_sigma'])
        if not 'shapelet_beta' in self.kwargs_fixed_else:
            mean.append(kwarg_mean_else['shapelet_beta'])
            sigma.append(kwarg_mean_else['shapelet_beta_sigma'])

        if self.kwargs_options.get('add_clump', False):
            if not 'theta_E_clump' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['theta_E_clump'])
                sigma.append(kwarg_mean_else['theta_E_clump_sigma'])
            if not 'r_trunc' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['r_trunc'])
                sigma.append(kwarg_mean_else['r_trunc_sigma'])
            if not 'x_clump' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['x_clump'])
                sigma.append(kwarg_mean_else['x_clump_sigma'])
            if not 'y_clump' in self.kwargs_fixed_else:
                mean.append(kwarg_mean_else['y_clump'])
                sigma.append(kwarg_mean_else['y_clump_sigma'])
        if self.kwargs_options.get('psf_iteration', False):
            if not 'point_amp' in self.kwargs_fixed_else:
                for i in range(1, len(kwarg_mean_else['point_amp'])):
                    mean.append(kwarg_mean_else['point_amp'][i])
                    sigma.append(kwarg_mean_else['point_amp_sigma'])
        return mean, sigma

    def param_bounds(self):
        """

        :return: hard bounds on the parameter space
        """
        #inizialize lower and upper limit arrays
        low = []
        high = []

        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_lens:
                low.append(0)
                high.append(1000)
            if not 'sigma_x' in self.kwargs_fixed_lens:
                low.append(-10)
                high.append(10)
            if not 'sigma_y' in self.kwargs_fixed_lens:
                low.append(-10)
                high.append(10)

        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS'  or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed_lens:
                low.append(0.001)
                high.append(10)
            if not 'gamma' in self.kwargs_fixed_lens:
                low.append(1.)
                high.append(2.85)
            if not 'q' in self.kwargs_fixed_lens or not 'phi_G' in self.kwargs_fixed_lens:
                low.append(-0.8)
                high.append(0.8)
                low.append(-0.8)
                high.append(0.8)

        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed_lens:
                low.append(-5)
                high.append(5)
            if not 'rho0' in self.kwargs_fixed_lens:
                low.append(-5)
                high.append(5)
            if not 'r200' in self.kwargs_fixed_lens:
                low.append(-5)
                high.append(5)
            if not 'center_x_nfw' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)
            if not 'center_y_nfw' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)

        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed_lens:
                low.append(-10)
                high.append(1)
            if not 'center_x_sis' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)
            if not 'center_y_sis' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)
        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed_lens:
                low.append(-10)
                high.append(5)
            if not 'gamma_spp' in self.kwargs_fixed_lens:
                low.append(1.45)
                high.append(2.85)
            if not 'center_x_spp' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)
            if not 'center_y_spp' in self.kwargs_fixed_lens:
                low.append(-3)
                high.append(3)

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed_lens:
                low.append(0.1)
                high.append(3.)
            if not 'coeffs' in self.kwargs_fixed_lens:
                num_coeffs = self.kwargs_options['num_shapelet_lens']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        num_coeffs -= 6
                    elif self.num_images == 2:
                        num_coeffs -= 3
                low += [-5]*num_coeffs
                high += [5]*num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed_lens:
                low.append(-2)
                high.append(2)
            if not 'center_y_shape' in self.kwargs_fixed_lens:
                low.append(-2)
                high.append(2)

        if self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed_lens:
                low.append(0)
                high.append(10)
            if not 'phi_dipole' in self.kwargs_fixed_else and self.kwargs_options['phi_dipole_decoupling'] is True:
                low.append(-np.pi)
                high.append(+np.pi)

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed_lens:
                low.append(-20)
                high.append(20)
            if not 'center_y' in self.kwargs_fixed_lens:
                low.append(-20)
                high.append(20)

        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_source:
                low.append(-3)
                high.append(3)
            if not 'sigma_x' in self.kwargs_fixed_source:
                low.append(-3)
                high.append(1)
            if not 'sigma_y' in self.kwargs_fixed_source:
                low.append(-3)
                high.append(1)

        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed_source:
                low.append(0)
                high.append(100)
            if not 'n_sersic' in self.kwargs_fixed_source:
                low.append(0.2)
                high.append(30)
            if not 'R_sersic' in self.kwargs_fixed_source: #acctually log(k**-2 n gamma(2*n))
                low.append(0.01)
                high.append(100)
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed_source or not 'q' in self.kwargs_fixed_source:
                    low.append(-0.8)
                    high.append(0.8)
                    low.append(-0.8)
                    high.append(0.8)
        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed_source:
                low.append(-10)
                high.append(10)
            if not 'center_y' in self.kwargs_fixed_source:
                low.append(-10)
                high.append(10)

        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_sersic' in self.kwargs_fixed_lens_light:
                low.append(0)
                high.append(100)
            if not 'n_sersic' in self.kwargs_fixed_lens_light:
                low.append(0.2)
                high.append(30)
            if not 'R_sersic' in self.kwargs_fixed_lens_light:
                low.append(0.01)
                high.append(30)
            if not 'center_x' in self.kwargs_fixed_lens_light:
                low.append(-10)
                high.append(10)
            if not 'center_y' in self.kwargs_fixed_lens_light:
                low.append(-10)
                high.append(10)
        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if not 'phi_G' in self.kwargs_fixed_lens_light or not 'q' in self.kwargs_fixed_lens_light:
                    low.append(-0.8)
                    high.append(0.8)
                    low.append(-0.8)
                    high.append(0.8)
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_2' in self.kwargs_fixed_lens_light:
                low.append(0)
                high.append(100)
            if not 'R_2' in self.kwargs_fixed_lens_light:
                low.append(0.01)
                high.append(30)
            if not 'n_2' in self.kwargs_fixed_lens_light:
                low.append(0.2)
                high.append(30)
            if not 'center_x_2' in self.kwargs_fixed_lens_light:
                low.append(-10)
                high.append(10)
            if not 'center_y_2' in self.kwargs_fixed_lens_light:
                low.append(-10)
                high.append(10)
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if not 'Rb' in self.kwargs_fixed_lens_light:
                low.append(0.01)
                high.append(30)
            if not 'gamma' in self.kwargs_fixed_lens_light:
                low.append(-3)
                high.append(3)
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed_lens_light:
                low.append(0)
                high.append(100)
            if not 'R_3' in self.kwargs_fixed_lens_light:
                low.append(0.01)
                high.append(10)
            if not 'n_3' in self.kwargs_fixed_lens_light:
                low.append(0.5)
                high.append(30)

        if not 'ra_pos' in self.kwargs_fixed_else:
            pos_low = -10
            pos_high = 10
            for i in range(self.num_images):
                low.append(pos_low)
                high.append(pos_high)
        if not 'dec_pos' in self.kwargs_fixed_else:
            pos_low = -10
            pos_high = 10
            for i in range(self.num_images):
                low.append(pos_low)
                high.append(pos_high)
        if self.kwargs_options.get('image_plane_source', False):
            if not 'source_pos_image_ra' in self.kwargs_fixed_else:
                low.append(-10)
                high.append(10)
            if not 'source_pos_image_dec' in self.kwargs_fixed_else:
                low.append(-10)
                high.append(10)
        if self.external_shear:
            if not 'gamma1' in self.kwargs_fixed_else or not 'gamma2' in self.kwargs_fixed_else:
                low.append(-0.8)
                high.append(0.8)
                low.append(-0.8)
                high.append(0.8)
        if self.foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed_else or not 'gamma2_foreground' in self.kwargs_fixed_else:
                low.append(-0.8)
                high.append(0.8)
                low.append(-0.8)
                high.append(0.8)
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed_else:
                low.append(0)
                high.append(10000)
        if not 'shapelet_beta' in self.kwargs_fixed_else:
            low.append(0.01)
            high.append(1)
        if self.kwargs_options.get('add_clump', False):
            if not 'theta_E_clump' in self.kwargs_fixed_else:
                low.append(0)
                high.append(1)
            if not 'r_trunc' in self.kwargs_fixed_else:
                low.append(0.001)
                high.append(1)
            if not 'x_clump' in self.kwargs_fixed_else:
                low.append(-10)
                high.append(10)
            if not 'y_clump' in self.kwargs_fixed_else:
                low.append(-10)
                high.append(10)
        if self.kwargs_options.get('psf_iteration', False):
            if not 'point_amp' in self.kwargs_fixed_else:
                for i in range(3):
                    low.append(0)
                    high.append(10)
        return np.asarray(low), np.asarray(high)


    def num_param(self):
        """

        :return: number of parameters involved (int)
        """
        num = 0
        list = []
        if self.kwargs_options['lens_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_lens:
                num+=1
                list.append('amp_lens')
            if not 'sigma_x' in self.kwargs_fixed_lens:
                num+=1
                list.append('sigma_x_lens')
            if not 'sigma_y' in self.kwargs_fixed_lens:
                num+=1
                list.append('sigma_y_lens')
        elif self.kwargs_options['lens_type'] == 'SPEP' or self.kwargs_options['lens_type'] == 'ELLIPSE' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_NFW' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SIS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E' in self.kwargs_fixed_lens:
                num+=1
                list.append('theta_E_lens')
            if not 'gamma' in self.kwargs_fixed_lens:
                num+=1
                list.append('gamma_lens')
            if not 'q' in self.kwargs_fixed_lens or not 'phi_G' in self.kwargs_fixed_lens:
                num+=2
                list.append('e1_lens')
                list.append('e2_lens')
        if self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SPEP_NFW':
            if not 'Rs' in self.kwargs_fixed_lens:
                num+=1
                list.append('Rs_nfw')
            if not 'rho0' in self.kwargs_fixed_lens:
                num+=1
                list.append('rho0_nfw')
            if not 'r200' in self.kwargs_fixed_lens:
                num+=1
                list.append('r200_nfw')
            if not 'center_x_nfw' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_x_nfw')
            if not 'center_y_nfw' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_y_nfw')
        elif self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPEP_SIS':
            if not 'theta_E_sis' in self.kwargs_fixed_lens:
                num+=1
                list.append('theta_E_sis')
            if not 'center_x_sis' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_x_sis')
            if not 'center_y_sis' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_y_sis')
        elif self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'theta_E_spp' in self.kwargs_fixed_lens:
                num+=1
                list.append('theta_E_spp')
            if not 'gamma_spp' in self.kwargs_fixed_lens:
                num+=1
                list.append('gamma_spp')
            if not 'center_x_spp' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_x_spp')
            if not 'center_y_spp' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_y_spp')

        if self.kwargs_options['lens_type'] == 'SHAPELETS_POLAR' or self.kwargs_options['lens_type'] == 'SPEP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'beta' in self.kwargs_fixed_lens:
                num+=1
                list.append('beta_lens')
            if not 'coeffs' in self.kwargs_fixed_lens:
                num_coeffs = self.kwargs_options['num_shapelet_lens']
                if self.solver_type == 'SHAPELETS':
                    if self.num_images == 4:
                        num_coeffs -= 6
                    elif self.num_images == 2:
                        num_coeffs -= 3
                num += num_coeffs
                list += ['coeff']*num_coeffs
            if not 'center_x_shape' in self.kwargs_fixed_lens:
                num += 1
                list.append('center_x_lens_shape')
            if not 'center_y_shape' in self.kwargs_fixed_lens:
                num += 1
                list.append('center_y_lens_shape')

        if self.kwargs_options['lens_type']  == 'SPEP_SPP_DIPOLE' or self.kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            if not 'coupling' in self.kwargs_fixed_lens:
                num += 1
                list.append('coupling')
            if not 'phi_dipole' in self.kwargs_fixed_else and self.kwargs_options['phi_dipole_decoupling'] is True:
                num += 1
                list.append('phi_dipole')

        if not(self.kwargs_options['lens_type'] == 'INTERPOL' or self.kwargs_options['lens_type'] == 'NFW' or self.kwargs_options['lens_type'] == 'SIS' or self.kwargs_options['lens_type'] == 'SPP' or self.kwargs_options['lens_type'] == 'NONE' or self.kwargs_options['lens_type'] == 'NONE'):
            if not 'center_x' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_x_lens')
            if not 'center_y' in self.kwargs_fixed_lens:
                num+=1
                list.append('center_y_lens')

        if self.kwargs_options['source_type'] == 'GAUSSIAN':
            if not 'amp' in self.kwargs_fixed_source:
                num += 1
                list.append('amp_source')
            if not 'sigma_x' in self.kwargs_fixed_source:
                num += 1
                list.append('sigma_x_source')
            if not 'sigma_y' in self.kwargs_fixed_source:
                num += 1
                list.append('sigma_y_source')
        elif self.kwargs_options['source_type'] == 'SERSIC' or self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
            if not 'I0_sersic' in self.kwargs_fixed_source:
                num += 1
                list.append('I0_sersic_source')
            if not 'n_sersic' in self.kwargs_fixed_source:
                num += 1
                list.append('n_sersic_source')
            if not 'R_sersic' in self.kwargs_fixed_source:
                num += 1
                list.append('R_sersic_source')
            if self.kwargs_options['source_type'] == 'SERSIC_ELLIPSE':
                if not 'phi_G' in self.kwargs_fixed_source or not 'q' in self.kwargs_fixed_source:
                    num += 2
                    list.append('e1_source')
                    list.append('e2_source')

        if not (self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False)) and not self.kwargs_options.get('image_plane_source', False):
            if not 'center_x' in self.kwargs_fixed_source:
                num += 1
                list.append('center_x_source')
            if not 'center_y' in self.kwargs_fixed_source:
                num += 1
                list.append('center_y_source')

        if self.kwargs_options['lens_light_type'] == 'SERSIC' or self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_sersic' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('I0_sersic_lens_light')
            if not 'n_sersic' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('n_sersic_lens_light')
            if not 'R_sersic' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('R_sersic_lens_light')
            if not 'center_x' in self.kwargs_fixed_lens_light:
                num+=1
                list.append('center_x_lens_light')
            if not 'center_y' in self.kwargs_fixed_lens_light:
                num+=1
                list.append('center_y_lens_light')
        if self.kwargs_options['lens_light_type'] == 'SERSIC_ELLIPSE' or self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
                if not 'phi_G' in self.kwargs_fixed_lens_light or not 'q' in self.kwargs_fixed_lens_light:
                    num += 2
                    list.append('e1_lens_light')
                    list.append('e2_lens_light')
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_SERSIC' or self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC' or self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_2' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('I2_lens_light')
            if not 'R_2' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('R_2_lens_light')
            if not 'n_2' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('n_2_lens_light')
            if not 'center_x_2' in self.kwargs_fixed_lens_light:
                num+=1
                list.append('center_x_2_lens_light')
            if not 'center_y_2' in self.kwargs_fixed_lens_light:
                num+=1
                list.append('center_y_2_lens_light')
        if self.kwargs_options['lens_light_type'] == 'DOUBLE_CORE_SERSIC':
            if not 'Rb' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('Rb_lens_light')
            if not 'gamma' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('gamma_lens_light')
        if self.kwargs_options['lens_light_type'] == 'TRIPPLE_SERSIC':
            if not 'I0_3' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('I0_3_lens_light')
            if not 'R_3' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('R_3_lens_light')
            if not 'n_3' in self.kwargs_fixed_lens_light:
                num += 1
                list.append('n_3_lens_light')

        if not 'ra_pos' in self.kwargs_fixed_else:
            num += self.num_images  # Warning: must be 4 point source positions!!!
            for i in range(self.num_images):
                list.append('ra_pos')
        if not 'dec_pos' in self.kwargs_fixed_else:
            num += self.num_images
            for i in range(self.num_images):
                list.append('dec_pos')
        if self.kwargs_options.get('image_plane_source', False):
            if not 'source_pos_image_ra' in self.kwargs_fixed_else:
                num += 1
                list.append('source_pos_image_ra')
            if not 'source_pos_image_dec' in self.kwargs_fixed_else:
                num += 1
                list.append('source_pos_image_dec')
        if self.external_shear:
            if not 'gamma1' in self.kwargs_fixed_else or not 'gamma2' in self.kwargs_fixed_else:
                num += 2
                list.append('shear_1')
                list.append('shear_2')
        if self.foreground_shear:
            if not 'gamma1_foreground' in self.kwargs_fixed_else or not 'gamma2_foreground' in self.kwargs_fixed_else:
                num += 2
                list.append('shear_foreground_1')
                list.append('shear_foreground_2')
        if self.kwargs_options.get('time_delay', False) is True:
            if not 'delay_dist' in self.kwargs_fixed_else:
                num += 1
                list.append('delay_dist')
        if not 'shapelet_beta' in self.kwargs_fixed_else:
            num += 1
            list.append('shapelet_beta')
        if self.kwargs_options.get('add_clump', False):
            if not 'theta_E_clump' in self.kwargs_fixed_else:
                num += 1
                list.append('theta_E_clump')
            if not 'r_trunc' in self.kwargs_fixed_else:
                num += 1
                list.append('r_trunc')
            if not 'x_clump' in self.kwargs_fixed_else:
                num += 1
                list.append('x_clump')
            if not 'y_clump' in self.kwargs_fixed_else:
                num += 1
                list.append('y_clump')
        if self.kwargs_options.get('psf_iteration', False):
            if not 'point_amp' in self.kwargs_fixed_else:
                num += 3
                for i in range(3):
                    list.append('point_amp')
        return num, list

    def _update_spep(self, kwargs_lens, x):
        """

        :param x: 1d array with spep parameters [phi_E, gamma, q, phi_G, center_x, center_y]
        :return: updated kwargs of lens parameters
        """
        [theta_E, e1, e2, center_x, center_y, non_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        kwargs_lens['theta_E'] = theta_E
        kwargs_lens['phi_G'] = phi_G
        kwargs_lens['q'] = q
        kwargs_lens['center_x'] = center_x
        kwargs_lens['center_y'] = center_y
        return kwargs_lens

    def _update_spep2(self, kwargs_lens, x):
        """

        :param x: 1d array with spep parameters [phi_E, gamma, q, phi_G, center_x, center_y]
        :return: updated kwargs of lens parameters
        """
        [center_x, center_y] = x
        kwargs_lens['center_x'] = center_x
        kwargs_lens['center_y'] = center_y
        return kwargs_lens

    def _update_coeffs(self, kwargs_lens, x):
        [c00, c10, c01, c20, c11, c02] = x
        coeffs = list(kwargs_lens['coeffs'])
        coeffs[0: 6] = [0, c10, c01, c20, c11, c02]
        kwargs_lens['coeffs'] = coeffs
        return kwargs_lens

    def _update_coeffs2(self, kwargs_lens, x):
        [c10, c01] = x
        coeffs = list(kwargs_lens['coeffs'])
        coeffs[1:3] = [c10, c01]
        kwargs_lens['coeffs'] = coeffs
        return kwargs_lens

    def get_all_params(self, args):
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.getParams(args)
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.update_kwargs(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else

    def update_kwargs(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
        if self.kwargs_options.get('solver', False):
            if self.foreground_shear:
                f_x_shear1, f_y_shear1 = self.makeImage.LensModel.shear.derivatives(kwargs_else['ra_pos'], kwargs_else['dec_pos'], e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
                x_ = kwargs_else['ra_pos'] - f_x_shear1
                y_ = kwargs_else['dec_pos'] - f_y_shear1
            else:
                x_, y_ = kwargs_else['ra_pos'], kwargs_else['dec_pos']
            if self.solver_type == 'SPEP' or self.solver_type == 'SPEMD':
                e1, e2 = util.phi_q2_elliptisity(kwargs_lens['phi_G'], kwargs_lens['q'])
                if self.num_images == 4:
                    init = np.array([kwargs_lens['theta_E'], e1, e2,
                            kwargs_lens['center_x'], kwargs_lens['center_y'], 0])  # sub-clump parameters to solve for
                    kwargs_lens['theta_E'] = 0
                    ra_sub, dec_sub = self.makeImage.LensModel.alpha(x_, y_, kwargs_else, **kwargs_lens)
                    x = self.constraints.get_param(x_, y_, ra_sub, dec_sub, init, {'gamma': kwargs_lens['gamma']})
                    kwargs_lens = self._update_spep(kwargs_lens, x)
                elif self.num_images == 2:
                    init = np.array([kwargs_lens['center_x'], kwargs_lens['center_y']])  # sub-clump parameters to solve for
                    theta_E = kwargs_lens['theta_E']
                    kwargs_lens['theta_E'] = 0
                    ra_sub, dec_sub = self.makeImage.LensModel.alpha(x_, y_, kwargs_else, **kwargs_lens)
                    x = self.constraints.get_param(x_, y_, ra_sub, dec_sub, init, {'gamma': kwargs_lens['gamma'],
                                'theta_E': theta_E, 'e1': e1, 'e2': e2})
                    kwargs_lens = self._update_spep2(kwargs_lens, x)
                else:
                    raise ValueError("%s number of images is not valid. Use 2 or 4!" % self.num_images)
            elif self.kwargs_options.get('solver_type', 'SPEP') == 'SHAPELETS':
                ra_sub, dec_sub = self.makeImage.LensModel.alpha(x_, y_, kwargs_else, **kwargs_lens)
                if self.num_images == 4:
                    init = [0, 0, 0, 0, 0, 0]
                    x = self.constraints.get_param(x_, y_, ra_sub, dec_sub, init, {'beta': kwargs_lens['beta'], 'center_x': kwargs_lens['center_x_shape'], 'center_y': kwargs_lens['center_y_shape']})
                    kwargs_lens = self._update_coeffs(kwargs_lens, x)
                elif self.num_images == 2:
                    init = [0, 0]
                    x = self.constraints.get_param(x_, y_, ra_sub, dec_sub, init, {'beta': kwargs_lens['beta'], 'center_x': kwargs_lens['center_x_shape'], 'center_y': kwargs_lens['center_y_shape']})
                    kwargs_lens = self._update_coeffs2(kwargs_lens, x)
            elif self.kwargs_options.get('solver_type', 'SPEP') == 'NONE':
                pass
        if self.kwargs_options.get('solver', False) or self.kwargs_options.get('fix_source', False) or self.kwargs_options.get('image_plane_source', False):
            if self.kwargs_options.get('image_plane_source', False):
                x_mapped, y_mapped = self.makeImage.mapping_IS(kwargs_else['source_pos_image_ra'], kwargs_else['source_pos_image_dec'], kwargs_else, **kwargs_lens)
                kwargs_source['center_x'] = x_mapped
                kwargs_source['center_y'] = y_mapped
            else:
                x_mapped, y_mapped = self.makeImage.mapping_IS(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_else, **kwargs_lens)
                #kwargs_source['center_x'] = np.mean(x_mapped)
                #kwargs_source['center_y'] = np.mean(y_mapped)
                kwargs_source['center_x'] = x_mapped[0]
                kwargs_source['center_y'] = y_mapped[0]
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else
