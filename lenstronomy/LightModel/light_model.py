__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the light models

import numpy as np
import copy


class LightModel(object):
    """
    class to handle source and lens light models
    """
    def __init__(self, light_model_list, deflection_scaling_list=None, source_redshift_list=None, smoothing=0.0000001):
        """

        :param light_model_list: list of light models
        :param deflection_scaling_list: list of floats, rescales the original reduced deflection angles from the lens model
        to enable different models to be placed at different optical (redshift) distances. None means they are all
        :param source_redshift_list: list of redshifts of the model components
        :param smoothing: smoothing factor for certain models (deprecated)
        """
        self.profile_type_list = light_model_list
        self.deflection_scaling_list = deflection_scaling_list
        self.redshift_list = source_redshift_list
        self.func_list = []
        for profile_type in light_model_list:
            if profile_type == 'GAUSSIAN':
                from lenstronomy.LightModel.Profiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif profile_type == 'GAUSSIAN_ELLIPSE':
                from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse
                self.func_list.append(GaussianEllipse())
            elif profile_type == 'MULTI_GAUSSIAN':
                from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
                self.func_list.append(MultiGaussian())
            elif profile_type == 'MULTI_GAUSSIAN_ELLIPSE':
                from lenstronomy.LightModel.Profiles.gaussian import MultiGaussianEllipse
                self.func_list.append(MultiGaussianEllipse())
            elif profile_type == 'SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import Sersic
                self.func_list.append(Sersic(smoothing=smoothing))
            elif profile_type == 'SERSIC_ELLIPSE':
                from lenstronomy.LightModel.Profiles.sersic import SersicElliptic
                self.func_list.append(SersicElliptic(smoothing=smoothing))
            elif profile_type == 'CORE_SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import CoreSersic
                self.func_list.append(CoreSersic(smoothing=smoothing))
            elif profile_type == 'SHAPELETS':
                from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
                self.func_list.append(ShapeletSet())
            elif profile_type == 'SHAPELETS_POLAR':
                from lenstronomy.LightModel.Profiles.shapelets_polar import ShapeletSetPolar
                self.func_list.append(ShapeletSetPolar(exponential=False))
            elif profile_type == 'SHAPELETS_POLAR_EXP':
                from lenstronomy.LightModel.Profiles.shapelets_polar import ShapeletSetPolar
                self.func_list.append(ShapeletSetPolar(exponential=True))
            elif profile_type == 'HERNQUIST':
                from lenstronomy.LightModel.Profiles.hernquist import Hernquist
                self.func_list.append(Hernquist())
            elif profile_type == 'HERNQUIST_ELLIPSE':
                from lenstronomy.LightModel.Profiles.hernquist import HernquistEllipse
                self.func_list.append(HernquistEllipse())
            elif profile_type == 'PJAFFE':
                from lenstronomy.LightModel.Profiles.p_jaffe import PJaffe
                self.func_list.append(PJaffe())
            elif profile_type == 'PJAFFE_ELLIPSE':
                from lenstronomy.LightModel.Profiles.p_jaffe import PJaffe_Ellipse
                self.func_list.append(PJaffe_Ellipse())
            elif profile_type == 'UNIFORM':
                from lenstronomy.LightModel.Profiles.uniform import Uniform
                self.func_list.append(Uniform())
            elif profile_type == 'POWER_LAW':
                from lenstronomy.LightModel.Profiles.power_law import PowerLaw
                self.func_list.append(PowerLaw())
            elif profile_type == 'NIE':
                from lenstronomy.LightModel.Profiles.nie import NIE
                self.func_list.append(NIE())
            elif profile_type == 'CHAMELEON':
                from lenstronomy.LightModel.Profiles.chameleon import Chameleon
                self.func_list.append(Chameleon())
            elif profile_type == 'DOUBLE_CHAMELEON':
                from lenstronomy.LightModel.Profiles.chameleon import DoubleChameleon
                self.func_list.append(DoubleChameleon())
            elif profile_type == 'TRIPLE_CHAMELEON':
                from lenstronomy.LightModel.Profiles.chameleon import TripleChameleon
                self.func_list.append(TripleChameleon())
            elif profile_type == 'INTERPOL':
                from lenstronomy.LightModel.Profiles.interpolation import Interpol
                self.func_list.append(Interpol())
            else:
                raise ValueError('Warning! No light model of type', profile_type, ' found!')

    def param_name_list(self):
        """
        returns the list of all parameter names

        :return: list of list of strings (for each light model separately)
        """
        name_list = []
        for func in self.func_list:
            name_list.append(func.param_names)
        return name_list

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        flux = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if k is None or k == i:
                out = np.array(func.function(x, y, **kwargs_list[i]), dtype=float)
                flux += out
        return flux

    def light_3d(self, r, kwargs_list, k=None):
        """
        computes 3d density at radius r
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        r = np.array(r, dtype=float)
        flux = np.zeros_like(r)
        for i, func in enumerate(self.func_list):
            if k is None or k == i:
                kwargs = {k: v for k, v in kwargs_list[i].items() if not k in ['center_x', 'center_y']}
                if self.profile_type_list[i] in ['HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE',
                                                     'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'MULTI_GAUSSIAN',
                                                     'MULTI_GAUSSIAN_ELLIPSE', 'POWER_LAW']:
                    flux += func.light_3d(r, **kwargs)
                else:
                    raise ValueError('Light model %s does not support a 3d light distribution!'
                                         % self.profile_type_list[i])
        return flux

    def functions_split(self, x, y, kwargs_list, k=None):
        """

        :param x:
        :param y:
        :param kwargs_list:
        :return:
        """
        response = []
        n = 0
        for i, model in enumerate(self.profile_type_list):
            if k is None or k == i:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE',
                             'PJAFFE_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                             'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                    new = {'amp': 1}
                    kwargs_new = kwargs_list[i].copy()
                    kwargs_new.update(new)
                    response += [self.func_list[i].function(x, y, **kwargs_new)]
                    n += 1
                elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                    num = len(kwargs_list[i]['amp'])
                    new = {'amp': np.ones(num)}
                    kwargs_new = kwargs_list[i].copy()
                    kwargs_new.update(new)
                    response += self.func_list[i].function_split(x, y, **kwargs_new)
                    n += num
                elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                    kwargs = kwargs_list[i]
                    n_max = kwargs['n_max']
                    if model in ['SHAPELETS_POLAR_EXP']:
                        num_param = int((n_max+1)**2)
                    else:
                        num_param = int((n_max + 1) * (n_max + 2) / 2)
                    new = {'amp': np.ones(num_param)}
                    kwargs_new = kwargs_list[i].copy()
                    kwargs_new.update(new)
                    response += self.func_list[i].function_split(x, y, **kwargs_new)
                    n += num_param
                else:
                    raise ValueError('model type %s not valid!' % model)
        return response, n

    def num_param_linear(self, kwargs_list=None):
        """

        :return: number of linear basis set coefficients
        """
        n = 0
        for i, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE',
                             'PJAFFE_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                             'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                n += 1
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num = len(kwargs_list[i]['sigma'])
                n += num
            elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                n_max = kwargs_list[i]['n_max']
                if model in ['SHAPELETS_POLAR_EXP']:
                    num_param = int((n_max+1)**2)
                else:
                    num_param = int((n_max + 1) * (n_max + 2) / 2)
                n += num_param
            else:
                raise ValueError('model type %s not valid!' % model)
        return n

    def update_linear(self, param, i, kwargs_list):
        """

        :param param:
        :param i:
        :param kwargs_list:
        :return:
        """
        for k, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                         'DOUBLE_CHAMELEON', 'TRIPLE_CHAMELEON', 'UNIFORM', 'INTERPOL']:
                kwargs_list[k]['amp'] = param[i]
                i += 1
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num_param = len(kwargs_list[k]['sigma'])
                kwargs_list[k]['amp'] = param[i:i + num_param]
                i += num_param
            elif model in ['SHAPELETS', 'SHAPELETS_POLAR', 'SHAPELETS_POLAR_EXP']:
                n_max = kwargs_list[k]['n_max']
                if model in ['SHAPELETS_POLAR_EXP']:
                    num_param = int((n_max+1)**2)
                else:
                    num_param = int((n_max + 1) * (n_max + 2) / 2)
                kwargs_list[k]['amp'] = param[i:i+num_param]
                i += num_param
            else:
                raise ValueError('model type %s not valid!' % model)
        return kwargs_list, i

    def re_normalize_flux(self, kwargs_list, norm_factor=1):
        """

        :param kwargs_list: list of keyword arguments
        :param norm_factor: float, multiplicative factor to rescale the amplitude parameters
        :return: new updated kwargs_list
        """
        kwargs_list_copy = copy.deepcopy(kwargs_list)
        kwargs_list_new = []
        for k, model in enumerate(self.profile_type_list):
            kwargs_list_k = kwargs_list_copy[k]
            if 'amp' in kwargs_list_k:
                kwargs_list_k['amp'] *= norm_factor
            kwargs_list_new.append(kwargs_list_k)
        return kwargs_list_new

    def total_flux(self, kwargs_list, norm=False, k=None):
        """
        Computes the total flux of each individual light profile. This allows to estimate the total flux as
        well as lenstronomy amp to magnitude conversions. Not all models are supported

        :param kwargs_list: list of keyword arguments corresponding to the light profiles. The 'amp' parameter can be missing.
        :param norm: bool, if True, computes the flux for amp=1
        :param k: int, if set, only evaluates the specific light model
        :return: list of (total) flux values attributed to each profile
        """
        norm_flux_list = []
        for i, model in enumerate(self.profile_type_list):
            if k is None or k == i:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'INTERPOL', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE',
                             'MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                    kwargs_new = kwargs_list[i].copy()
                    if norm is True:
                        if model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                            new = {'amp': np.array(kwargs_new['amp'])/kwargs_new['amp'][0]}
                        else:
                            new = {'amp': 1}
                        kwargs_new.update(new)
                    norm_flux = self.func_list[i].total_flux(**kwargs_new)
                    norm_flux_list.append(norm_flux)
                else:
                    raise ValueError("profile %s does not support flux normlization." % model)
                #  TODO implement total flux for e.g. 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE',
                    # 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON', 'DOUBLE_CHAMELEON' ,
                # 'TRIPLE_CHAMELEON', 'UNIFORM'
        return norm_flux_list
