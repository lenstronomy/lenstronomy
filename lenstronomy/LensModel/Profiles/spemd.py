__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['SPEMD']


class SPEMD(LensProfileBase):
    """
    class for smooth power law ellipse mass density profile. This class effectively performs the FASTELL calculations
    by Renan Barkana. The parameters are changed and represent a spherically averaged Einstein radius an a logarithmic
    3D mass profile slope.

    The Einstein ring parameter converts to the definition used by GRAVLENS as follow:
    (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]


    FASTELL has the following defintions:
    The parameters are position (x1,x2), overall factor
    (q), power (gam), axis ratio (arat) which is <=1, core radius
    squared (s2), and the output potential (phi).
    The projected mass density distribution, in units of the
    critical density, is kappa(x1,x2)=q [u2+s2]^(-gam), where
    u2=[x1^2+x2^2/(arat^2)].
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 's_scale', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 's_scale': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 's_scale': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self, suppress_fastell=False):
        """

        """
        try:
            from fastell4py import fastell4py
            self._fastell4py_bool = True
            self.fastell4py = fastell4py
        except:
            self._fastell4py_bool = False
            if suppress_fastell:
                ImportWarning("module fastell4py not installed. You can get it from here: "
                              "https://github.com/sibirrer/fastell4py "
                              "Make sure you have a fortran compiler such that the installation works properly.")
                Warning("SPEMD model outputs are replaced by zeros as fastell4py package is not installed!")
            else:
                raise ImportError("module fastell4py not installed. You can get it from here: "
                                  "https://github.com/sibirrer/fastell4py "
                                  "Make sure you have a fortran compiler such that the installation works properly.")
        super(SPEMD, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale in the center of the profile (angle)
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: lensing potential
        """
        x1, x2, q_fastell, gam, s2, q, phi_G = self.param_transform(x, y, theta_E, gamma, e1, e2, s_scale, center_x,
                                                                    center_y)
        compute_bool = self._parameter_constraints(q_fastell, gam, s2, q)
        if self._fastell4py_bool and self.is_not_empty(x1, x2) and compute_bool:
            potential = self.fastell4py.ellipphi(x1, x2, q_fastell, gam, arat=q, s2=s2)
            n = len(np.atleast_1d(x))
            if n <= 1:
                if np.shape(x) == ():
                    return np.array(potential[0])
        else:
            potential = np.zeros_like(x1)
        return potential

    def derivatives(self, x, y, theta_E, gamma, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale in the center of the profile
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: deflection angles alpha_x, alpha_y
        """
        x1, x2, q_fastell, gam, s2, q, phi_G = self.param_transform(x, y, theta_E, gamma, e1, e2, s_scale, center_x,
                                                                    center_y)
        compute_bool = self._parameter_constraints(q_fastell, gam, s2, q)
        if self._fastell4py_bool and self.is_not_empty(x1, x2) and compute_bool:
            f_x_prim, f_y_prim = self.fastell4py.fastelldefl(x1, x2, q_fastell, gam, arat=q, s2=s2)
        else:
            f_x_prim, f_y_prim =  np.zeros_like(x1), np.zeros_like(x1)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale in the center of the profile
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: Hessian components f_xx, f_xy, f_yx, f_yy
        """
        x1, x2, q_fastell, gam, s2, q, phi_G = self.param_transform(x, y, theta_E, gamma, e1, e2, s_scale, center_x, center_y)
        compute_bool = self._parameter_constraints(q_fastell, gam, s2, q)
        if self._fastell4py_bool and self.is_not_empty(x1, x2) and compute_bool:
            f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = self.fastell4py.fastellmag(x1, x2, q_fastell, gam,
                                                                                             arat=q, s2=s2)
            n = len(np.atleast_1d(x))
            if n <= 1:
                if np.shape(x) == ():
                    f_xx_prim, f_yy_prim, f_xy_prim = np.array(f_xx_prim[0]), np.array(f_yy_prim[0]), np.array(
                        f_xy_prim[0])
        else:
            f_xx_prim, f_yy_prim, f_xy_prim = np.zeros_like(x1), np.zeros_like(x1), np.zeros_like(x1)
        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    def param_transform(self, x, y, theta_E, gamma, e1, e2, s_scale, center_x=0, center_y=0):
        """
        transforms parameters in the format of fastell4py

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale in the center of the profile
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: x-rotated, y-rotated, q_fastell, gam, s2, q, phi_G
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x = np.array(x)
        y = np.array(y)
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam, s2 = self.convert_params(theta_E, gamma, q, s_scale)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi * x_shift + sin_phi * y_shift
        x2 = -sin_phi * x_shift + cos_phi * y_shift
        return x1, x2, q_fastell, gam, s2, q, phi_G

    @staticmethod
    def convert_params(theta_E, gamma, q, s_scale):
        """
        converts parameter defintions into quantities used by the FASTELL fortran library

        :param theta_E: Einstein radius
        :param gamma: 3D power-law slope of mass profile
        :param q: axis ratio minor/major
        :param s_scale: float, smoothing scale in the core
        :return: pre-factors to SPEMP profile for FASTELL
        """
        gam = (gamma-1)/2.
        q_fastell = (3-gamma)/2. * (theta_E ** 2 / q) ** gam
        s2 = s_scale ** 2
        return q_fastell, gam, s2

    @staticmethod
    def is_not_empty(x1, x2):
        """
        Check if float or not an empty array
        :return: True if x1 and x2 are either floats/ints or an non-empty array, False if e.g. objects are []
        :rtype: bool
        """
        assert type(x1) == type(x2)

        if isinstance(x1, (list, tuple, np.ndarray)):
            if len(x1) != 0 and len(x2) != 0:
                return True
            else:
                return False
        else:
            return True

    @staticmethod
    def _parameter_constraints(q_fastell, gam, s2, q):
        """
        sets bounds to parameters due to numerical stability

        FASTELL has the following defintions:
        The parameters are position (x1,x2), overall factor
        (q), power (gam), axis ratio (arat) which is <=1, core radius
        squared (s2), and the output potential (phi).
        The projected mass density distribution, in units of the
        critical density, is kappa(x1,x2)=q [u2+s2]^(-gam), where
        u2=[x1^2+x2^2/(arat^2)].

        :param q_fastell: float, normalization of lens model, q_fastell = (3-gamma)/2. * (theta_E ** 2 / q) ** gam
        :param gam: float, slope parameter, gam = (gamma-1)/2.
        :param q: axis ratio
        :param s2: square of smoothing scale of the core
        :return: bool of whether or not to let the fastell provide to be evaluated or instead return zero(s)
        """
        if q_fastell < 0 or s2 < 0.0000000000001 or q > 1 or q < 0.01 or gam > 0.999 or gam < 0.001 or \
                not np.isfinite(q_fastell):
            return False
        return True
