__author__ = "sibirrer"

__all__ = ["SpecialParam"]

import numpy as np
from .param_group import ModelParamGroup, SingleParam, ArrayParam


# ==================================== #
# == Defining individual parameters == #
# ==================================== #


class DdtSamplingParam(SingleParam):
    """
    Time delay parameter
    """

    param_names = ["D_dt"]
    _kwargs_lower = {"D_dt": 0}
    _kwargs_upper = {"D_dt": 100000}


class DdSamplingParam(SingleParam):
    """
    Deflector distance parameter
    """

    param_names = ["D_d"]
    _kwargs_lower = {"D_d": 0}
    _kwargs_upper = {"D_d": 100000}


class BetaAnisotropyParam(SingleParam):
    """
    Cylindrical anisotropy parameter
    """

    param_names = ["b_ani"]
    _kwargs_lower = {"b_ani": -1}
    _kwargs_upper = {"b_ani": 1}


class InclinationParam(SingleParam):
    """
    Inclination parameter (radians)
    """

    param_names = ["incli"]
    _kwargs_lower = {"incli": 0}
    _kwargs_upper = {"incli": np.pi / 2}


class SourceSizeParam(SingleParam):
    """
    Source size parameter
    """

    param_names = ["source_size"]
    _kwargs_lower = {"source_size": 0}
    _kwargs_upper = {"source_size": 1}


class SourceGridOffsetParam(SingleParam):
    """
    Source grid offset, both x and y.
    """

    param_names = ["delta_x_source_grid", "delta_y_source_grid"]
    _kwargs_lower = {"delta_x_source_grid": -100, "delta_y_source_grid": -100}
    _kwargs_upper = {"delta_x_source_grid": 100, "delta_y_source_grid": 100}


class MassScalingParam(ArrayParam):
    """
    Mass scaling. Can scale the masses of arbitrary subsets of lens models
    """

    _kwargs_lower = {"scale_factor": 0}
    _kwargs_upper = {"scale_factor": 1000}

    def __init__(self, num_scale_factor):
        super().__init__(on=int(num_scale_factor) > 0)
        self.param_names = {"scale_factor": int(num_scale_factor)}


class PointSourceOffsetParam(ArrayParam):
    """
    Point source offset, both x and y
    """

    _kwargs_lower = {"delta_x_image": -1, "delta_y_image": -1}
    _kwargs_upper = {"delta_x_image": 1, "delta_y_image": 1}

    def __init__(self, offset, num_images):
        super().__init__(on=offset and (int(num_images) > 0))
        self.param_names = {
            "delta_x_image": int(num_images),
            "delta_y_image": int(num_images),
        }


class Tau0ListParam(ArrayParam):
    """
    Optical depth renormalization parameters
    """

    _kwargs_lower = {"tau0_list": 0}
    _kwargs_upper = {"tau0_list": 1000}

    def __init__(self, num_tau0):
        super().__init__(on=int(num_tau0) > 0)
        self.param_names = {"tau0_list": int(num_tau0)}


class ZSamplingParam(ArrayParam):
    """
    Redshift sampling.
    """

    _kwargs_lower = {"z_sampling": 0}
    _kwargs_upper = {"z_sampling": 1000}

    def __init__(self, num_z_sampling):
        super().__init__(on=int(num_z_sampling) > 0)
        self.param_names = {"z_sampling": int(num_z_sampling)}


class GeneralScalingParam(ArrayParam):
    """
    General lens scaling.

    For each scaled lens parameter, adds a `{param}_scale_factor` and
    `{param}_scale_pow` special parameter, and updates the scaled param
    as `param = param_scale_factor * param**param_scale_pow`.
    """

    def __init__(self, params: dict):
        # params is a dictionary
        self.param_names = {}
        self._kwargs_lower = {}
        self._kwargs_upper = {}

        super().__init__(params)
        if not self.on:
            return

        for name, array in params.items():
            num_param = np.max(array)

            if num_param > 0:
                fac_name = f"{name}_scale_factor"
                self.param_names[fac_name] = num_param
                self._kwargs_lower[fac_name] = 0
                self._kwargs_upper[fac_name] = 1000

                pow_name = f"{name}_scale_pow"
                self.param_names[pow_name] = num_param
                self._kwargs_lower[pow_name] = -10
                self._kwargs_upper[pow_name] = 10


# ======================================== #
# == All together: Composing into class == #
# ======================================== #


class SpecialParam(object):
    """
    class that handles special parameters that are not directly part of a specific model component.
    These includes cosmology relevant parameters, astrometric errors and overall scaling parameters.
    """

    def __init__(
        self,
        Ddt_sampling=False,
        mass_scaling=False,
        num_scale_factor=1,
        general_scaling_params=None,
        kwargs_fixed=None,
        kwargs_lower=None,
        kwargs_upper=None,
        point_source_offset=False,
        source_size=False,
        num_images=0,
        num_tau0=0,
        num_z_sampling=0,
        source_grid_offset=False,
        kinematic_sampling=False,
    ):
        """

        :param Ddt_sampling: bool, if True, samples the time-delay distance D_dt (in units of Mpc)
        :param mass_scaling: bool, if True, samples a mass scaling factor between different profiles
        :param num_scale_factor: int, number of independent mass scaling factors being sampled
        :param kwargs_fixed: keyword arguments, fixed parameters during sampling
        :param kwargs_lower: keyword arguments, lower bound of parameters being sampled
        :param kwargs_upper: keyword arguments, upper bound of parameters being sampled
        :param point_source_offset: bool, if True, adds relative offsets ot the modeled image positions relative to the
         time-delay and lens equation solver
        :param num_images: number of point source images such that the point source offset parameters match their
         numbers
        :param source_size: bool, if True, samples a source size parameters to be evaluated in the flux ratio likelihood
        :param num_tau0: integer, number of different optical depth re-normalization factors
        :param num_z_sampling: integer, number of different lens redshifts to be sampled
        :param source_grid_offset: bool, if True, samples two parameters (x, y) for the offset of the pixelated source
         plane grid coordinates.
         Warning: this is only defined for pixel-based source modelling (e.g. 'SLIT_STARLETS' light profile)
        :param kinematic_sampling: bool, if True, samples the kinematic parameters b_ani, incli, with cosmography
         D_dt (overrides _D_dt_sampling) and Dd
        """

        self._D_dt_sampling = DdtSamplingParam(Ddt_sampling or kinematic_sampling)

        self._D_d_sampling = DdSamplingParam(kinematic_sampling)
        self._b_ani_sampling = BetaAnisotropyParam(kinematic_sampling)
        self._incli_sampling = InclinationParam(kinematic_sampling)
        if not mass_scaling:
            num_scale_factor = 0
        self._mass_scaling = MassScalingParam(num_scale_factor)

        self._general_scaling = GeneralScalingParam(general_scaling_params or dict())

        if point_source_offset:
            self._point_source_offset = PointSourceOffsetParam(True, num_images)
        else:
            self._point_source_offset = PointSourceOffsetParam(False, 0)
        self._source_size = SourceSizeParam(source_size)
        self._tau0 = Tau0ListParam(num_tau0)
        self._z_sampling = ZSamplingParam(num_z_sampling)
        self._source_grid_offset = SourceGridOffsetParam(source_grid_offset)

        if kwargs_fixed is None:
            kwargs_fixed = {}
        self._kwargs_fixed = kwargs_fixed

        if kwargs_lower is None:
            kwargs_lower = {}
            for group in self._param_groups:
                kwargs_lower = dict(kwargs_lower, **group.kwargs_lower)
        if kwargs_upper is None:
            kwargs_upper = {}
            for group in self._param_groups:
                kwargs_upper = dict(kwargs_upper, **group.kwargs_upper)

        self.lower_limit = kwargs_lower
        self.upper_limit = kwargs_upper

    def get_params(self, args, i, impose_bound=False):
        """

        :param args: argument list
        :param i: integer, list index to start the read out for this class
        :param impose_bound: bool, if True, imposes the lower and upper limits on the sampled parameters
        :return: keyword arguments related to args, index after reading out arguments of this class
        """
        if impose_bound:
            result = ModelParamGroup.compose_get_params(
                self._param_groups,
                args,
                i,
                kwargs_fixed=self._kwargs_fixed,
                kwargs_lower=self.lower_limit,
                kwargs_upper=self.upper_limit,
            )
        else:
            result = ModelParamGroup.compose_get_params(
                self._param_groups, args, i, kwargs_fixed=self._kwargs_fixed
            )
        return result

    def set_params(self, kwargs_special):
        """

        :param kwargs_special: keyword arguments with parameter settings
        :return: argument list of the sampled parameters extracted from kwargs_special
        """
        return ModelParamGroup.compose_set_params(
            self._param_groups, kwargs_special, kwargs_fixed=self._kwargs_fixed
        )

    def num_param(self):
        """

        :return: integer, number of free parameters sampled (and managed) by this class, parameter names (list of strings)
        """
        return ModelParamGroup.compose_num_params(
            self._param_groups, kwargs_fixed=self._kwargs_fixed
        )

    @property
    def _param_groups(self):
        return [
            self._D_dt_sampling,
            self._D_d_sampling,
            self._b_ani_sampling,
            self._incli_sampling,
            self._mass_scaling,
            self._general_scaling,
            self._point_source_offset,
            self._source_size,
            self._tau0,
            self._z_sampling,
            self._source_grid_offset,
        ]
