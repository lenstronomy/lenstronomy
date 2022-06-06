__author__ = 'dgilman'

from lenstronomy.Util.param_util import shear_cartesian2polar, shear_polar2cartesian
from lenstronomy.Util.param_util import ellipticity2phi_q
import numpy as np

# class ParamClass(object):
#
#     """
#     This class handles converting a dict of parameter values into an array of parameter values to
#     feed to the optimizer. This class can be user specified to implement any combination of lens models with arbitrary
#     constraints on their parameters. Three examples are shown below for power law like mass models.
#
#     This class, if specified by user, must have the methods "kwargs_to_args" and "args_to_kwargs" (see below)
#     """
#     def __init__(self, kwargs_lens_init, **kwargs):
#
#         """
#
#         :param kwargs_lens_init: the initial kwargs_lens before optimizing
#         :param args: any other args for the particular optimization routine corresponding to this class
#         """
#
#         to_vary_index is the number of lens models with parameters that are being optimized. For power law + shear
#         it equals 2, for example. The lens models being optimized should always come first in the lens_model_list
#
#         pass
#
#     @property
#     def to_vary_index(self):
#         return integer
#
#     def bounds(self, scale):
#
#         pass
#
#     @staticmethod
#     def kwargs_to_args(kwargs):
#
#         pass
#
#     def args_to_kwargs(self, args):
#
#         pass

class Hessian(object):

    def __init__(self, kwargs_lens_init):
        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    def bounds(self):

        raise Exception('PSO is not intended for this class so no bounds are needed.')

    @property
    def to_vary_index(self):
        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 1

    @staticmethod
    def kwargs_to_args(kwargs):
        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        fxx = kwargs[0]['f_xx']
        fxy = kwargs[0]['f_xy']
        fyx = kwargs[0]['f_yx']
        fyy = kwargs[0]['f_yy']

        args = (fxx, fxy, fyx, fyy)

        return args

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (fxx, fxy, fyx, fyy) = args
        self.kwargs_lens[0]['f_xx'] = fxx
        self.kwargs_lens[0]['f_xy'] = fxy
        self.kwargs_lens[0]['f_yx'] = fyx
        self.kwargs_lens[0]['f_yy'] = fyy

        return self.kwargs_lens

class CurvedArcBase(object):

    def __init__(self, kwargs_lens_init):
        """
        # ['tangential_stretch', 'radial_stretch', 'curvature', 'dtan_dtan', 'direction', 'center_x', 'center_y']
        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    @property
    def to_vary_index(self):
        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 1

class CurvedArcFixedCurvature(CurvedArcBase):

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        radial_stretch = kwargs[0]['radial_stretch']
        tangential_stretch = kwargs[0]['tangential_stretch']
        dtan_dtan = kwargs[0]['dtan_dtan']
        direction = kwargs[0]['direction']

        args = (tangential_stretch, radial_stretch, dtan_dtan, direction)

        return args

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            d_radial_stretch = 2.
            d_tangential_stretch = 2.
            d_direction = np.pi / 10
            d_dtan_dtan = 0.5

        else:
            d_radial_stretch = 10
            d_tangential_stretch = 10
            d_direction = np.pi / 2
            d_dtan_dtan = 10.0

        shifts = np.array([d_tangential_stretch, d_radial_stretch, d_dtan_dtan, d_direction]) * scale

        low, high = np.empty_like(shifts), np.empty_like(shifts)

        for i in range(0, len(shifts)):
            low[i] = args[i] / shifts[i]
            high[i] = args[i] * shifts[i]

        return low, high

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (tangential_stretch, radial_stretch, dtan_dtan, direction) = args
        curvature = self.kwargs_lens[0]['curvature']
        center_x = self.kwargs_lens[0]['center_x']
        center_y = self.kwargs_lens[0]['center_y']

        kwargs = {'radial_stretch': radial_stretch, 'tangential_stretch': tangential_stretch, 'dtan_dtan': dtan_dtan,
         'curvature': curvature, 'direction': direction, 'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs

        return self.kwargs_lens

class CurvedArcFree(CurvedArcBase):

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        radial_stretch = kwargs[0]['radial_stretch']
        tangential_stretch = kwargs[0]['tangential_stretch']
        curvature = kwargs[0]['curvature']
        direction = kwargs[0]['direction']
        dtan_dtan = kwargs[0]['dtan_dtan']

        args = (radial_stretch, tangential_stretch, curvature, direction, dtan_dtan)

        return args

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            d_radial_stretch = 2.
            d_tangential_stretch = 2.
            d_direction = np.pi / 10
            d_curvature = 2.
            d_dtan_dtan = 0.5

        else:
            d_radial_stretch = 10
            d_tangential_stretch = 10
            d_direction = np.pi / 2
            d_curvature = 10.
            d_dtan_dtan = 10.

        shifts = np.array([d_radial_stretch, d_tangential_stretch, d_curvature, d_direction, d_dtan_dtan]) * scale

        low, high = np.empty_like(shifts), np.empty_like(shifts)

        for i in range(0, len(shifts)):
            low[i] = args[i] / shifts[i]
            high[i] = args[i] * shifts[i]

        return low, high

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (radial_stretch, tangential_stretch, curvature, direction, dtan_dtan) = args

        center_x = self.kwargs_lens[0]['center_x']
        center_y = self.kwargs_lens[0]['center_y']

        kwargs = {'radial_stretch': radial_stretch, 'tangential_stretch': tangential_stretch, 'dtan_dtan': dtan_dtan,
         'curvature': curvature, 'direction': direction, 'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs

        return self.kwargs_lens

class CurvedArcFixedCurvatureDirection(CurvedArcBase):

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        radial_stretch = kwargs[0]['radial_stretch']
        tangential_stretch = kwargs[0]['tangential_stretch']
        dtan_dtan = kwargs[0]['dtan_dtan']

        args = (radial_stretch, tangential_stretch, dtan_dtan)

        return args

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            d_radial_stretch = 2.
            d_tangential_stretch = 2.
            d_dtan_dtan = 0.5

        else:
            d_radial_stretch = 10
            d_tangential_stretch = 10
            d_dtan_dtan = 10

        shifts = np.array([d_radial_stretch, d_tangential_stretch, d_dtan_dtan]) * scale
        low, high = np.empty_like(shifts), np.empty_like(shifts)
        for i in range(0, len(shifts)):
            low[i] = args[i] / shifts[i]
            high[i] = args[i] * shifts[i]
        return low, high

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (radial_stretch, tangential_stretch) = args
        curvature = self.kwargs_lens[0]['curvature']
        center_x = self.kwargs_lens[0]['center_x']
        center_y = self.kwargs_lens[0]['center_y']
        direction = self.kwargs_lens[0]['direction']
        dtan_dtan = self.kwargs_lens[0]['dtan_dtan']

        kwargs = {'radial_stretch': radial_stretch, 'tangential_stretch': tangential_stretch, 'dtan_dtan': dtan_dtan,
         'curvature': curvature, 'direction': direction, 'center_x': center_x, 'center_y': center_y}

        self.kwargs_lens[0] = kwargs

        return self.kwargs_lens

class ShiftLensModelParamManager(object):

    def __init__(self, kwargs_lens_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    @property
    def to_vary_index(self):
        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 1

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            dx = 0.1
            dy = 0.1

        else:
            dx = 0.5
            dy = 0.5

        shifts = np.array([dx, dy])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        alpha_x_shift, alpha_y_shift = kwargs[0]['alpha_x_shift'], kwargs[1]['alpha_y_shift']
        args = (alpha_x_shift, alpha_y_shift)
        return args

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        (alpha_x_shift, alpha_y_shift) = args
        self.kwargs_lens[0] = {'alpha_x': alpha_x_shift, 'alpha_y': alpha_y_shift}
        return self.kwargs_lens

class PowerLawParamManager(object):

    """
    Base class for handling the translation between key word arguments and parameter arrays for
    EPL mass models. This class is intended for use in modeling galaxy-scale lenses
    """

    def __init__(self, kwargs_lens_init):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        """

        self.kwargs_lens = kwargs_lens_init

    def param_chi_square_penalty(self, args):

        return 0.

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 2 because the first three lens models
        are EPL and SHEAR, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 2

    def bounds(self, re_optimize, scale=1.):

        """
        Sets the low/high parameter bounds for the particle swarm optimization

        NOTE: The low/high values specified here are intended for galaxy-scale lenses. If you want to use this
        for a different size system you should create a new ParamClass with different settings

        :param re_optimize: keep a narrow window around each parameter
        :param scale: scales the size of the uncertainty window
        :return:
        """

        args = self.kwargs_to_args(self.kwargs_lens)

        if re_optimize:
            thetaE_shift = 0.005
            center_shift = 0.01
            e_shift = 0.05
            g_shift = 0.025

        else:
            thetaE_shift = 0.25
            center_shift = 0.2
            e_shift = 0.2
            g_shift = 0.05

        shifts = np.array([thetaE_shift, center_shift, center_shift, e_shift, e_shift, g_shift, g_shift])

        low = np.array(args) - shifts * scale
        high = np.array(args) + shifts * scale
        return low, high

    @staticmethod
    def kwargs_to_args(kwargs):

        """

        :param kwargs: keyword arguments corresponding to the lens model parameters being optimized
        :return: array of lens model parameters
        """

        thetaE = kwargs[0]['theta_E']
        center_x = kwargs[0]['center_x']
        center_y = kwargs[0]['center_y']
        e1 = kwargs[0]['e1']
        e2 = kwargs[0]['e2']
        g1 = kwargs[1]['gamma1']
        g2 = kwargs[1]['gamma2']

        args = (thetaE, center_x, center_y, e1, e2, g1, g2)
        return args


class PowerLawFreeShear(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope allowed to vary
    """
    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters
        """

        gamma = self.kwargs_lens[0]['gamma']
        kwargs_epl = {'theta_E': args[0], 'center_x': args[1], 'center_y': args[2],
                      'e1': args[3], 'e2': args[4], 'gamma': gamma}

        kwargs_shear = {'gamma1': args[5], 'gamma2': args[6]}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        return self.kwargs_lens


class PowerLawFixedShear(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear with every parameter except the power law slope AND the
    shear strength allowed to vary. The user should specify shear_strengh in the args_param_class keyword when
    creating the Optimizer class
    """

    def __init__(self, kwargs_lens_init, shear_strength):

        """

        :param kwargs_lens_init: the initial kwargs_lens before optimizing
        :param shear_strength: the strenght of the external shear to be kept fixed
        """
        self._shear_strength = shear_strength

        super(PowerLawFixedShear, self).__init__(kwargs_lens_init)

    def args_to_kwargs(self, args):

        """

        :param args: array of lens model parameters
        :return: dictionary of lens model parameters with fixed shear = shear_strength
        """

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}

        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        return self.kwargs_lens


class PowerLawFreeShearMultipole(PowerLawParamManager):

    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope and multipole moment free to vary. The mass centroid and orientation of the multipole term are
    fixed to that of the EPL profile

    """

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 3

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args

        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}
        kwargs_shear = {'gamma1': g1, 'gamma2': g2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['phi_m'] = phi

        return self.kwargs_lens


class PowerLawFixedShearMultipole(PowerLawFixedShear):
    """
    This class implements a fit of EPL + external shear + a multipole term with every parameter except the
    power law slope, shear strength, and multipole moment free to vary. The mass centroid and orientation of the
    multipole term are fixed to that of the EPL profile
    """

    @property
    def to_vary_index(self):

        """
        The number of lens models being varied in this routine. This is set to 3 because the first three lens models
        are EPL, SHEAR, and MULTIPOLE, and their parameters are being optimized.

        The kwargs_list is split at to to_vary_index with indicies < to_vary_index accessed in this class,
        and lens models with indicies > to_vary_index kept fixed.

        Note that this requires a specific ordering of lens_model_list
        :return:
        """

        return 3

    def args_to_kwargs(self, args):

        (thetaE, center_x, center_y, e1, e2, g1, g2) = args
        gamma = self.kwargs_lens[0]['gamma']

        kwargs_epl = {'theta_E': thetaE, 'center_x': center_x, 'center_y': center_y,
                      'e1': e1, 'e2': e2, 'gamma': gamma}

        phi, _ = shear_cartesian2polar(g1, g2)
        gamma1, gamma2 = shear_polar2cartesian(phi, self._shear_strength)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        self.kwargs_lens[0] = kwargs_epl
        self.kwargs_lens[1] = kwargs_shear

        self.kwargs_lens[2]['center_x'] = center_x
        self.kwargs_lens[2]['center_y'] = center_y
        phi, _ = ellipticity2phi_q(e1, e2)
        self.kwargs_lens[2]['phi_m'] = phi

        return self.kwargs_lens
