__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the lens light

import numpy as np


class LensLightModel(object):

    def __init__(self, light_model_list=['NONE']):
        self.lightModel = LightModel(light_model_list, smoothing=0.001)

    def surface_brightness(self, x, y, kwargs_lens_light_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.lightModel.surface_brightness(x, y, kwargs_lens_light_list, k=k)


class SourceModel(object):

    def __init__(self, light_model_list=['NONE']):
        self.lightModel = LightModel(light_model_list, smoothing=0.00001)

    def surface_brightness(self, x, y, kwargs_source_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.lightModel.surface_brightness(x, y, kwargs_source_list, k=k)


class LightModel(object):
    """
    class to handle source and lens light models
    """
    def __init__(self, profile_type_list, smoothing=0.001):
        self.profile_type_list = profile_type_list
        self.func_list = []
        self.valid_list = []
        for profile_type in profile_type_list:
            valid = True
            if profile_type == 'GAUSSIAN':
                from lenstronomy.LightModel.Profiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif profile_type == 'MULTI_GAUSSIAN':
                from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
                self.func_list.append(MultiGaussian())
            elif profile_type == 'SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import Sersic
                self.func_list.append(Sersic(smoothing=smoothing))
            elif profile_type == 'SERSIC_ELLIPSE':
                from lenstronomy.LightModel.Profiles.sersic import Sersic_elliptic
                self.func_list.append(Sersic_elliptic(smoothing=smoothing))
            elif profile_type == 'DOUBLE_SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import DoubleSersic
                self.func_list.append(DoubleSersic(smoothing=smoothing))
            elif profile_type == 'CORE_SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import CoreSersic
                self.func_list.append(CoreSersic(smoothing=smoothing))
            elif profile_type == 'DOUBLE_CORE_SERSIC':
                from lenstronomy.LightModel.Profiles.sersic import DoubleCoreSersic
                self.func_list.append(DoubleCoreSersic(smoothing=smoothing))
            elif profile_type == 'BULDGE_DISK':
                from lenstronomy.LightModel.Profiles.sersic import BuldgeDisk
                self.func_list.append(BuldgeDisk(smoothing=smoothing))
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
            elif profile_type == 'NONE':
                valid = False
            else:
                valid = False
                raise ValueError('Warning! No light model of type', profile_type, ' found!')
            self.valid_list.append(valid)

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        flux = np.zeros_like(x)
        for i, func in enumerate(self.func_list):
            if self.valid_list[i]:
                if k == None or k == i:
                    flux += func.function(x, y, **kwargs_list[i])
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
            if self.valid_list[i]:
                if k == None or k == i:
                    kwargs = {k: v for k, v in kwargs_list[i].items() if not k in ['center_x', 'center_y']}
                    if self.profile_type_list[i] in ['HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE', 'GAUSSIAN', 'MULTI_GAUSSIAN']:
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
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                new = {'I0_sersic': 1, 'I0_2': 1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += 2
            elif model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
                new = {'I0_sersic': 1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['BULDGE_DISK']:
                new = {'I0_b': 1, 'I0_d': 1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += 2
            elif model in ['HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE']:
                new = {'sigma0': 1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['GAUSSIAN']:
                new = {'amp':  1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['MULTI_GAUSSIAN']:
                num = len(kwargs_list[k]['amps'])
                new = {'amp': np.ones(num)}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += num
            elif model in ['SHAPELETS']:
                kwargs = kwargs_list[k]
                n_max = kwargs['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                new = {'amp': np.ones(num_param)}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += num_param
            elif model in ['UNIFORM']:
                new = {'mean':  1}
                kwargs_new = kwargs_list[k].copy()
                kwargs_new.update(new)
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['NONE']:
                pass
            else:
                raise ValueError('model type %s not valid!' % model)
        return response, n

    def re_normalize_flux(self, kwargs_list, norm_factor=1):
        """

        :param kwargs:
        :return:
        """
        for k, model in enumerate(self.profile_type_list):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_list[k]['I0_sersic'] *= norm_factor
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_list[k]['I0_2'] *= norm_factor
            if model in ['BULDGE_DISK']:
                kwargs_list[k]['I0_b'] *= norm_factor
                kwargs_list[k]['I0_d'] *= norm_factor
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_list[k]['sigma0'] *= norm_factor
            if model in ['GAUSSIAN', 'MULTI_GAUSSIAN']:
                kwargs_list[k]['amp'] *= norm_factor
            if model in ['SHAPELETS']:
                kwargs_list[k]['amp'] *= norm_factor
            if model in ['UNIFORM']:
                kwargs_list[k]['mean'] *= norm_factor
        return kwargs_list
