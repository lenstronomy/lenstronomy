__author__ = 'sibirrer'


import numpy as np
import astrofunc.util as util

from lenstronomy.MCMC.solver4point import Constraints
from lenstronomy.MCMC.solver2point import Constraints2
from lenstronomy.ImSim.make_image import MakeImage

from lenstronomy.Workflow.lens_param import LensParam
from lenstronomy.Workflow.source_param import SourceParam
from lenstronomy.Workflow.lens_light_param import LensLightParam
from lenstronomy.Workflow.else_param import ElseParam


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
        self.lensParams = LensParam(kwargs_options, kwargs_fixed_lens)
        self.souceParams = SourceParam(kwargs_options, kwargs_fixed_source)
        self.lensLightParams = LensLightParam(kwargs_options, kwargs_fixed_lens_light)
        self.elseParams = ElseParam(kwargs_options, kwargs_fixed_else)

    def getParams(self, args):
        """

        :param args: tuple of parameter values (float, strings, ...(
        :return: keyword arguments sorted
        """
        i = 0
        kwargs_lens, i = self.lensParams.getParams(args, i)
        kwargs_source, i = self.souceParams.getParams(args, i)
        kwargs_lens_light, i = self.lensLightParams.getParams(args, i)
        kwargs_else, i = self.elseParams.getParams(args, i)

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
        args = self.lensParams.setParams(kwargs_lens)
        args += self.souceParams.setParams(kwargs_source)
        args += self.lensLightParams.setParams(kwargs_lens_light)
        args += self.elseParams.setParams(kwargs_else)
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
        lens_fix = self.lensParams.add2fix(lens_fixed)
        source_fix = self.souceParams.add2fix(source_fixed)
        lens_light_fix = self.lensLightParams.add2fix(lens_light_fixed)
        else_fix = self.elseParams.add2fix(else_fixed)
        return lens_fix, source_fix, lens_light_fix, else_fix

    def param_init(self, kwarg_mean_lens, kwarg_mean_source, kwarg_mean_lens_light={}, kwarg_mean_else={}):
        """
        returns upper and lower bounds on the parameters used in the X2_chain function for MCMC/PSO starting
        bounds are defined relative to the catalogue level image called in the class Data
        might be migrated to the param class
        """
        #inizialize mean and sigma limit arrays
        mean, sigma = self.lensParams.param_init(kwarg_mean_lens)
        _mean, _sigma = self.souceParams.param_init(kwarg_mean_source)
        mean += _mean
        sigma += _sigma
        _mean, _sigma = self.lensLightParams.param_init(kwarg_mean_lens_light)
        mean += _mean
        sigma += _sigma
        _mean, _sigma = self.elseParams.param_init(kwarg_mean_else)
        mean += _mean
        sigma += _sigma
        return mean, sigma

    def param_bounds(self):
        """

        :return: hard bounds on the parameter space
        """
        #inizialize lower and upper limit arrays
        low, high = self.lensParams.param_bounds()
        _low, _high = self.souceParams.param_bound()
        low += _low
        high += _high
        _low, _high = self.lensLightParams.param_bound()
        low += _low
        high += _high
        _low, _high = self.elseParams.param_bound()
        low += _low
        high += _high
        return np.asarray(low), np.asarray(high)

    def num_param(self):
        """

        :return: number of parameters involved (int)
        """
        num, list = self.lensParams.num_param()
        _num, _list = self.souceParams.num_param()
        num += _num
        list += _list
        _num, _list = self.lensLightParams.num_param()
        num += _num
        list += _list
        _num, _list = self.elseParams.num_param()
        num += _num
        list += _list
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
                    kwargs_lens['theta_E'] = theta_E
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
