__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the lens light

import numpy as np


class LensLightModel(object):

    def __init__(self, kwargs_options):
        lens_light_model_list = kwargs_options.get('lens_light_model_list', ['NONE'])
        self.lightModel = LightModel(lens_light_model_list)

    def surface_brightness(self, x, y, kwargs_lens_light_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        return self.lightModel.surface_brightness(x, y, kwargs_lens_light_list, k=k)


class SourceModel(object):

    def __init__(self, kwargs_options):
        source_light_model_list = kwargs_options.get('source_light_model_list', ['NONE'])
        self.lightModel = LightModel(source_light_model_list)

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
    def __init__(self, profile_type_list):
        self.profile_type_list = profile_type_list
        self.func_list = []
        self.valid_list = []
        for profile_type in profile_type_list:
            valid = True
            if profile_type == 'GAUSSIAN':
                from astrofunc.LensingProfiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif profile_type == 'SERSIC':
                from astrofunc.LightProfiles.sersic import Sersic
                self.func_list.append(Sersic())
            elif profile_type == 'SERSIC_ELLIPSE':
                from astrofunc.LightProfiles.sersic import Sersic_elliptic
                self.func_list.append(Sersic_elliptic())
            elif profile_type == 'SHAPELETS':
                from astrofunc.LensingProfiles.shapelets import ShapeletSet
                self.func_list.append(ShapeletSet())
            elif profile_type == 'DOUBLE_SERSIC':
                from astrofunc.LightProfiles.sersic import DoubleSersic
                self.func_list.append(DoubleSersic())
            elif profile_type == 'CORE_SERSIC':
                from astrofunc.LightProfiles.sersic import CoreSersic
                self.func_list.append(CoreSersic())
            elif profile_type == 'DOUBLE_CORE_SERSIC':
                from astrofunc.LightProfiles.sersic import DoubleCoreSersic
                self.func_list.append(DoubleCoreSersic())
            elif profile_type == 'BULDGE_DISK':
                from astrofunc.LightProfiles.sersic import BuldgeDisk
                self.func_list.append(BuldgeDisk())
            elif profile_type == 'HERNQUIST':
                from astrofunc.LightProfiles.hernquist import Hernquist
                self.func_list.append(Hernquist())
            elif profile_type == 'HERNQUIST_ELLIPSE':
                from astrofunc.LightProfiles.hernquist import Hernquist_Ellipse
                self.func_list.append(Hernquist_Ellipse())
            elif profile_type == 'P_JAFFE':
                from astrofunc.LightProfiles.p_jaffe import PJaffe
                self.func_list.append(PJaffe())
            elif profile_type == 'P_JAFFE_ELLIPSE':
                from astrofunc.LightProfiles.p_jaffe import PJaffe_Ellipse
                self.func_list.append(PJaffe_Ellipse())
            elif profile_type == 'NONE':
                valid = False
            else:
                print('Warning! No light model of type', profile_type, ' found!')
                valid = False
            self.valid_list.append(valid)

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        flux = np.zeros(len(x))
        for i, func in enumerate(self.func_list):
            if self.valid_list[i]:
                if k == None or k == i:
                    flux += func.function(x, y, **kwargs_list[i])
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
                kwargs_new = dict(kwargs_list[k].items() + new.items())
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += 2
            elif model in ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC']:
                new = {'I0_sersic': 1}
                kwargs_new = dict(kwargs_list[k].items() + new.items())
                response += [self.func_list[k].function(x, y, **kwargs_new)]
                n += 1
            elif model in ['BULDGE_DISK']:
                new = {'I0_b': 1, 'I0_d': 1}
                kwargs_new = dict(kwargs_list[k].items() + new.items())
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += 2
            elif model in ['SHAPELETS']:
                kwargs = kwargs_list[k]
                n_max = kwargs['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                new = {'amp': np.ones(num_param)}
                kwargs_new = dict(kwargs.items() + new.items())
                response += self.func_list[k].function_split(x, y, **kwargs_new)
                n += num_param
            elif model in ['NONE']:
                pass
            else:
                raise ValueError('model type %s not valid!' % model)
        return response, n
