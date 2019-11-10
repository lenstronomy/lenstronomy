__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the light models

import numpy as np
from lenstronomy.Util.util import convert_bool_list


class LightModelBase(object):
    """
    class to handle source and lens light models
    """
    def __init__(self, light_model_list, smoothing=0.0000001):
        """

        :param light_model_list: list of light models
        :param deflection_scaling_list: list of floats, rescales the original reduced deflection angles from the lens model
        to enable different models to be placed at different optical (redshift) distances. None means they are all
        :param source_redshift_list: list of redshifts of the model components
        :param smoothing: smoothing factor for certain models (deprecated)
        """
        self.profile_type_list = light_model_list
        self.func_list = []
        for profile_type in light_model_list:
            if profile_type == 'GAUSSIAN':
                from lenstronomy.LightModel.Profiles.gaussian import Gaussian
                self.func_list.append(Gaussian())
            elif profile_type == 'GAUSSIAN_ELLIPSE':
                from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse
                self.func_list.append(GaussianEllipse())
            elif profile_type == 'ELLIPSOID':
                from lenstronomy.LightModel.Profiles.ellipsoid import Ellipsoid
                self.func_list.append(Ellipsoid())
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
        self._num_func = len(self.func_list)

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        flux = np.zeros_like(x)
        bool_list = self._bool_list(k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                out = np.array(func.function(x, y, **kwargs_list_standard[i]), dtype=float)
                flux += out
        return flux

    def light_3d(self, r, kwargs_list, k=None):
        """
        computes 3d density at radius r
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        r = np.array(r, dtype=float)
        flux = np.zeros_like(r)
        bool_list = self._bool_list(k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs = {k: v for k, v in kwargs_list_standard[i].items() if not k in ['center_x', 'center_y']}
                if self.profile_type_list[i] in ['HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE',
                                                     'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'MULTI_GAUSSIAN',
                                                     'MULTI_GAUSSIAN_ELLIPSE', 'POWER_LAW']:
                    flux += func.light_3d(r, **kwargs)
                else:
                    raise ValueError('Light model %s does not support a 3d light distribution!'
                                         % self.profile_type_list[i])
        return flux

    def total_flux(self, kwargs_list, norm=False, k=None):
        """
        Computes the total flux of each individual light profile. This allows to estimate the total flux as
        well as lenstronomy amp to magnitude conversions. Not all models are supported

        :param kwargs_list: list of keyword arguments corresponding to the light profiles. The 'amp' parameter can be missing.
        :param norm: bool, if True, computes the flux for amp=1
        :param k: int, if set, only evaluates the specific light model
        :return: list of (total) flux values attributed to each profile
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        norm_flux_list = []
        bool_list = self._bool_list(k=k)
        for i, model in enumerate(self.profile_type_list):
            if bool_list[i] is True:
                if model in ['SERSIC', 'SERSIC_ELLIPSE', 'INTERPOL', 'GAUSSIAN', 'GAUSSIAN_ELLIPSE',
                             'MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE']:
                    kwargs_new = kwargs_list_standard[i].copy()
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

    def _transform_kwargs(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list as parameterised models
        :return: keyword argument list as used in the individual models
        """
        return kwargs_list

    def _bool_list(self, k=None):
        """
        returns a bool list of the length of the lens models
        if k = None: returns bool list with True's
        if k is int, returns bool list with False's but k'th is True
        if k is a list of int, e.g. [0, 3, 5], returns a bool list with True's in the integers listed and False elsewhere
        if k is a boolean list, checks for size to match the numbers of models and returns it

        :param k: None, int, or list of ints
        :return: bool list
        """
        return convert_bool_list(n=self._num_func, k=k)
