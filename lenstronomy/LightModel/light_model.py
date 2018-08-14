__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the lens light

import numpy as np


class LightModel(object):
    """
    class to handle source and lens light models
    """
    def __init__(self, light_model_list, smoothing=0.0000001):
        self.profile_type_list = light_model_list
        self.func_list = []
        for profile_type in light_model_list:
            valid = True
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
                from lenstronomy.LightModel.Profiles.sersic import Sersic_elliptic
                self.func_list.append(Sersic_elliptic(smoothing=smoothing))
            elif profile_type == 'CORE_SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import CoreSersic
                self.func_list.append(CoreSersic(smoothing=smoothing))
            elif profile_type == 'SHAPELETS':
                from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
                self.func_list.append(ShapeletSet())
            elif profile_type == 'HERNQUIST':
                from lenstronomy.LightModel.Profiles.hernquist import Hernquist
                self.func_list.append(Hernquist())
            elif profile_type == 'HERNQUIST_ELLIPSE':
                from lenstronomy.LightModel.Profiles.hernquist import Hernquist_Ellipse
                self.func_list.append(Hernquist_Ellipse())
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

    def functions_split(self, x, y, kwargs_list):
        """

        :param x:
        :param y:
        :param kwargs_list:
        :return:
        """
        response = []
        n = 0
        for k, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE',
                         'PJAFFE_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON', 'DOUBLE_CHAMELEON', 'UNIFORM']:
                new = {'amp': 1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num = len(kwargs_list[k]['amp'])
                new = {'amp': np.ones(num)}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += num
            elif model in ['SHAPELETS']:
                kwargs = kwargs_list[k]
                n_max = kwargs['n_max']
                num_param = int((n_max + 1) * (n_max + 2) / 2)
                new = {'amp': np.ones(num_param)}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += num_param
            else:
                raise ValueError('model type %s not valid!' % model)
        return response, n

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
                         'DOUBLE_CHAMELEON', 'UNIFORM']:
                kwargs_list[k]['amp'] = param[i]
                i += 1
            elif model in ['MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                num_param = len(kwargs_list[k]['sigma'])
                kwargs_list[k]['amp'] = param[i:i + num_param]
                i += num_param
            elif model in ['SHAPELETS']:
                n_max = kwargs_list[k]['n_max']
                num_param = int((n_max + 1) * (n_max + 2) / 2)
                kwargs_list[k]['amp'] = param[i:i+num_param]
                i += num_param
            else:
                raise ValueError('model type %s not valid!' % model)
        return kwargs_list, i

    def re_normalize_flux(self, kwargs_list, norm_factor=1):
        """

        :param kwargs:
        :return:
        """
        kwargs_list_new = []
        for k, model in enumerate(self.profile_type_list):
            kwargs_list_k = kwargs_list[k].copy()
            if 'amp' in kwargs_list_k:
                kwargs_list_k['amp'] *= norm_factor
            kwargs_list_new.append(kwargs_list_k)
        return kwargs_list_new

    def check_positive_flux_profile(self, kwargs_list):
        pos_bool = True
        for k, model in enumerate(self.profile_type_list):
            if 'amp' in kwargs_list[k]:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE',
                         'HERNQUIST_ELLIPSE', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON',
                         'DOUBLE_CHAMELEON']:
                    if kwargs_list[k]['amp'] < 0:
                        pos_bool = False
                        break
        return pos_bool