__author__ = "sibirrer"

# this file contains a class which describes the surface brightness of the light models

import numpy as np
from lenstronomy.Util.util import convert_bool_list
from lenstronomy.Conf import config_loader

convention_conf = config_loader.conventions_conf()
sersic_major_axis_conf = convention_conf.get("sersic_major_axis", False)

__all__ = ["LightModelBase"]


_MODELS_SUPPORTED = [
    "GAUSSIAN",
    "GAUSSIAN_ELLIPSE",
    "ELLIPSOID",
    "MULTI_GAUSSIAN",
    "MULTI_GAUSSIAN_ELLIPSE",
    "SERSIC",
    "SERSIC_ELLIPSE",
    "SERSIC_ELLIPSE_Q_PHI",
    "CORE_SERSIC",
    "SHAPELETS",
    "SHAPELETS_POLAR",
    "SHAPELETS_POLAR_EXP",
    "SHAPELETS_ELLIPSE",
    "HERNQUIST",
    "HERNQUIST_ELLIPSE",
    "PJAFFE",
    "PJAFFE_ELLIPSE",
    "UNIFORM",
    "POWER_LAW",
    "NIE",
    "CHAMELEON",
    "DOUBLE_CHAMELEON",
    "TRIPLE_CHAMELEON",
    "INTERPOL",
    "SLIT_STARLETS",
    "SLIT_STARLETS_GEN2",
    "LINEAR",
    "LINEAR_ELLIPSE",
    "LINE_PROFILE",
]


class LightModelBase(object):
    """Class to handle source and lens light models."""

    def __init__(self, light_model_list, smoothing=0.001, sersic_major_axis=None):
        """

        :param light_model_list: list of light models
        :param smoothing: smoothing factor for certain models (deprecated)
        :param sersic_major_axis: boolean or None, if True, uses the semi-major axis as the definition of the Sersic
         half-light radius, if False, uses the product average of semi-major and semi-minor axis. If None, uses the
         convention in the lenstronomy yaml setting (which by default is =False)
        """
        self.profile_type_list = light_model_list
        self.func_list = []
        if sersic_major_axis is None:
            sersic_major_axis = sersic_major_axis_conf
        for profile_type in light_model_list:
            if profile_type == "GAUSSIAN":
                from lenstronomy.LightModel.Profiles.gaussian import Gaussian

                self.func_list.append(Gaussian())
            elif profile_type == "GAUSSIAN_ELLIPSE":
                from lenstronomy.LightModel.Profiles.gaussian import GaussianEllipse

                self.func_list.append(GaussianEllipse())
            elif profile_type == "ELLIPSOID":
                from lenstronomy.LightModel.Profiles.ellipsoid import Ellipsoid

                self.func_list.append(Ellipsoid())
            elif profile_type == "MULTI_GAUSSIAN":
                from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian

                self.func_list.append(MultiGaussian())
            elif profile_type == "MULTI_GAUSSIAN_ELLIPSE":
                from lenstronomy.LightModel.Profiles.gaussian import (
                    MultiGaussianEllipse,
                )

                self.func_list.append(MultiGaussianEllipse())
            elif profile_type == "SERSIC":
                from lenstronomy.LightModel.Profiles.sersic import Sersic

                self.func_list.append(Sersic(smoothing=smoothing))
            elif profile_type == "SERSIC_ELLIPSE":
                from lenstronomy.LightModel.Profiles.sersic import SersicElliptic

                self.func_list.append(
                    SersicElliptic(
                        smoothing=smoothing, sersic_major_axis=sersic_major_axis
                    )
                )
            elif profile_type == "SERSIC_ELLIPSE_Q_PHI":
                from lenstronomy.LightModel.Profiles.sersic import SersicElliptic_qPhi

                self.func_list.append(
                    SersicElliptic_qPhi(
                        smoothing=smoothing, sersic_major_axis=sersic_major_axis
                    )
                )
            elif profile_type == "CORE_SERSIC":
                from lenstronomy.LightModel.Profiles.sersic import CoreSersic

                self.func_list.append(
                    CoreSersic(smoothing=smoothing, sersic_major_axis=sersic_major_axis)
                )
            elif profile_type == "SHAPELETS":
                from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

                self.func_list.append(ShapeletSet())
            elif profile_type == "SHAPELETS_ELLIPSE":
                from lenstronomy.LightModel.Profiles.shapelets_ellipse import (
                    ShapeletSetEllipse,
                )

                self.func_list.append(ShapeletSetEllipse())
            elif profile_type == "SHAPELETS_POLAR":
                from lenstronomy.LightModel.Profiles.shapelets_polar import (
                    ShapeletSetPolar,
                )

                self.func_list.append(ShapeletSetPolar(exponential=False))
            elif profile_type == "SHAPELETS_POLAR_EXP":
                from lenstronomy.LightModel.Profiles.shapelets_polar import (
                    ShapeletSetPolar,
                )

                self.func_list.append(ShapeletSetPolar(exponential=True))
            elif profile_type == "HERNQUIST":
                from lenstronomy.LightModel.Profiles.hernquist import Hernquist

                self.func_list.append(Hernquist())
            elif profile_type == "HERNQUIST_ELLIPSE":
                from lenstronomy.LightModel.Profiles.hernquist import HernquistEllipse

                self.func_list.append(HernquistEllipse())
            elif profile_type == "PJAFFE":
                from lenstronomy.LightModel.Profiles.p_jaffe import PJaffe

                self.func_list.append(PJaffe())
            elif profile_type == "PJAFFE_ELLIPSE":
                from lenstronomy.LightModel.Profiles.p_jaffe import PJaffeEllipse

                self.func_list.append(PJaffeEllipse())
            elif profile_type == "UNIFORM":
                from lenstronomy.LightModel.Profiles.uniform import Uniform

                self.func_list.append(Uniform())
            elif profile_type == "POWER_LAW":
                from lenstronomy.LightModel.Profiles.power_law import PowerLaw

                self.func_list.append(PowerLaw())
            elif profile_type == "NIE":
                from lenstronomy.LightModel.Profiles.nie import NIE

                self.func_list.append(NIE())
            elif profile_type == "CHAMELEON":
                from lenstronomy.LightModel.Profiles.chameleon import Chameleon

                self.func_list.append(Chameleon())
            elif profile_type == "DOUBLE_CHAMELEON":
                from lenstronomy.LightModel.Profiles.chameleon import DoubleChameleon

                self.func_list.append(DoubleChameleon())
            elif profile_type == "TRIPLE_CHAMELEON":
                from lenstronomy.LightModel.Profiles.chameleon import TripleChameleon

                self.func_list.append(TripleChameleon())
            elif profile_type == "INTERPOL":
                from lenstronomy.LightModel.Profiles.interpolation import Interpol

                self.func_list.append(Interpol())
            elif profile_type == "SLIT_STARLETS":
                from lenstronomy.LightModel.Profiles.starlets import SLIT_Starlets

                self.func_list.append(
                    SLIT_Starlets(fast_inverse=True, second_gen=False)
                )
            elif profile_type == "SLIT_STARLETS_GEN2":
                from lenstronomy.LightModel.Profiles.starlets import SLIT_Starlets

                self.func_list.append(SLIT_Starlets(second_gen=True))
            elif profile_type == "LINEAR":
                from lenstronomy.LightModel.Profiles.linear import Linear

                self.func_list.append(Linear())
            elif profile_type == "LINEAR_ELLIPSE":
                from lenstronomy.LightModel.Profiles.linear import LinearEllipse

                self.func_list.append(LinearEllipse())
            elif profile_type == "LINE_PROFILE":
                from lenstronomy.LightModel.Profiles.lineprofile import LineProfile

                self.func_list.append(LineProfile())
            else:
                raise ValueError(
                    "No light model of type %s found! Supported are the following models: %s"
                    % (profile_type, _MODELS_SUPPORTED)
                )
        self._num_func = len(self.func_list)

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """
        :param x: coordinate in units of arcsec relative to the center of the image
        :type x: set or single 1d numpy array
        :param y: coordinate in units of arcsec relative to the center of the image
        :type y: set or single 1d numpy array
        :param kwargs_list: keyword argument list of light profile
        :param k: integer or list of integers for selecting subsets of light profiles
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        flux = np.zeros_like(x)
        bool_list = self._bool_list(k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                out = np.array(
                    func.function(x, y, **kwargs_list_standard[i]), dtype=float
                )
                flux += out
        return flux

    def light_3d(self, r, kwargs_list, k=None):
        """Computes 3d density at radius r (3D radius) such that integrated in
        projection in units of angle results in the projected surface brightness.

        :param r: 3d radius units of arcsec relative to the center of the light profile
        :param kwargs_list: keyword argument list of light profile
        :param k: integer or list of integers for selecting subsets of light profiles.
        :return: flux density
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        r = np.array(r, dtype=float)
        flux = np.zeros_like(r)
        bool_list = self._bool_list(k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                kwargs = {
                    k: v
                    for k, v in kwargs_list_standard[i].items()
                    if k not in ["center_x", "center_y"]
                }
                if self.profile_type_list[i] in [
                    "DOUBLE_CHAMELEON",
                    "CHAMELEON",
                    "HERNQUIST",
                    "HERNQUIST_ELLIPSE",
                    "PJAFFE",
                    "PJAFFE_ELLIPSE",
                    "GAUSSIAN",
                    "GAUSSIAN_ELLIPSE",
                    "MULTI_GAUSSIAN",
                    "MULTI_GAUSSIAN_ELLIPSE",
                    "NIE",
                    "POWER_LAW",
                    "TRIPLE_CHAMELEON",
                ]:
                    flux += func.light_3d(r, **kwargs)
                else:
                    raise ValueError(
                        "Light model %s does not support a 3d light distribution!"
                        % self.profile_type_list[i]
                    )
        return flux

    def total_flux(self, kwargs_list, norm=False, k=None):
        """Computes the total flux of each individual light profile. This allows to
        estimate the total flux as well as lenstronomy amp to magnitude conversions. Not
        all models are supported. The units are linked to the data to be modelled with
        associated noise properties (default is count/s).

        :param kwargs_list: list of keyword arguments corresponding to the light
            profiles. The 'amp' parameter can be missing.
        :param norm: bool, if True, computes the flux for amp=1
        :param k: int, if set, only evaluates the specific light model
        :return: list of (total) flux values attributed to each profile
        """
        kwargs_list_standard = self._transform_kwargs(kwargs_list)
        norm_flux_list = []
        bool_list = self._bool_list(k=k)
        for i, model in enumerate(self.profile_type_list):
            if bool_list[i] is True:
                if model in [
                    "SERSIC",
                    "SERSIC_ELLIPSE",
                    "INTERPOL",
                    "GAUSSIAN",
                    "GAUSSIAN_ELLIPSE",
                    "MULTI_GAUSSIAN",
                    "MULTI_GAUSSIAN_ELLIPSE",
                    "LINE_PROFILE",
                ]:
                    kwargs_new = kwargs_list_standard[i].copy()
                    if norm is True:
                        if model in ["MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE"]:
                            new = {
                                "amp": np.array(kwargs_new["amp"])
                                / kwargs_new["amp"][0]
                            }
                        else:
                            new = {"amp": 1}
                        kwargs_new.update(new)
                    norm_flux = self.func_list[i].total_flux(**kwargs_new)
                    norm_flux_list.append(norm_flux)
                else:
                    raise ValueError(
                        "profile %s does not support flux normlization." % model
                    )
                #  TODO implement total flux for e.g. 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE',
                # 'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 'POWER_LAW', 'NIE', 'CHAMELEON', 'DOUBLE_CHAMELEON' ,
                # 'TRIPLE_CHAMELEON', 'UNIFORM'
        return norm_flux_list

    def delete_interpol_caches(self):
        """Call the delete_cache method of INTERPOL profiles."""
        for i, model in enumerate(self.profile_type_list):
            if model in ["INTERPOL", "SLIT_STARLETS", "SLIT_STARLETS_GEN2"]:
                self.func_list[i].delete_cache()

    def _transform_kwargs(self, kwargs_list):
        """

        :param kwargs_list: keyword argument list as parameterised models
        :return: keyword argument list as used in the individual models
        """
        return kwargs_list

    def _bool_list(self, k=None):
        """Returns a bool list of the length of the lens models if k = None: returns
        bool list with True's if k is int, returns bool list with False's but k'th is
        True if k is a list of int, e.g. [0, 3, 5], returns a bool list with True's in
        the integers listed and False elsewhere if k is a boolean list, checks for size
        to match the numbers of models and returns it.

        :param k: None, int, or list of ints
        :return: bool list
        """
        return convert_bool_list(n=self._num_func, k=k)
