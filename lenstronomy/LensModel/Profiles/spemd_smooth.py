__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class SPEMD_SMOOTH(LensProfileBase):
    """
    class for smooth power law ellipse mass density profile
    The Einstein ring parameter converts to the definition used by GRAVLENS as follow:
    (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 's_scale', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 's_scale': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 's_scale': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        try:
            from fastell4py import fastell4py
            self._fastell4py_bool = True
            self.fastell4py = fastell4py
        except:
            self._fastell4py_bool = False
            print("module fastell4py not installed. You can get it from here: https://github.com/sibirrer/fastell4py")
        super(SPEMD_SMOOTH, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, s_scale, center_x=0, center_y=0):
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
        :return: lensing potential
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E, gamma, q, phi_G, s_scale = self._parameter_constraints(theta_E, gamma, q, phi_G, s_scale)
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(theta_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift
        if self._fastell4py_bool and self.is_not_empty(x1, x2):
            potential = self.fastell4py.ellipphi(x1, x2, q_fastell, gam, arat=q, s2=s_scale)
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
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x = np.array(x)
        y = np.array(y)
        theta_E, gamma, q, phi_G, s_scale = self._parameter_constraints(theta_E, gamma, q, phi_G, s_scale)
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(theta_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift

        if self._fastell4py_bool and self.is_not_empty(x1, x2):
            f_x_prim, f_y_prim = self.fastell4py.fastelldefl(x1, x2, q_fastell, gam, arat=q, s2=s_scale)
        else:
            f_x_prim, f_y_prim =  np.zeros_like(x1), np.zeros_like(x1)
        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim
        n = len(np.atleast_1d(x))
        #if n <= 1:
        #    if np.shape(x) == ():
        #        f_x, f_y = np.array(f_x[0]), np.array(f_y[0])
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
        :return: Hessian components f_xx, f_yy, f_xy
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E, gamma, q, phi_G, s_scale = self._parameter_constraints(theta_E, gamma, q, phi_G, s_scale)
        x = np.array(x)
        y = np.array(y)
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(theta_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift
        if self._fastell4py_bool and self.is_not_empty(x1, x2):
            f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = self.fastell4py.fastellmag(x1, x2, q_fastell, gam,
                                                                                             arat=q, s2=s_scale)
            n = len(np.atleast_1d(x))
            if n <= 1:
                if np.shape(x) == ():
                    f_xx_prim, f_yy_prim, f_xy_prim = np.array(f_xx_prim[0]), np.array(f_yy_prim[0]), np.array(
                        f_xy_prim[0])
        else:
            f_xx_prim, f_yy_prim, f_xy_prim =  np.zeros_like(x1), np.zeros_like(x1), np.zeros_like(x1)
        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    @staticmethod
    def convert_params(theta_E, gamma, q):
        """
        converts parameter defintions into quantities used by the FASTELL fortran library

        :param theta_E: Einstein radius
        :param gamma: 3D power-law slope of mass profile
        :param q: axis ratio minor/major
        :return: pre-factors to SPEMP profile for FASTELL
        """
        gam = (gamma-1)/2.
        q_fastell = (3-gamma)/2. * (theta_E ** 2 / q) ** gam
        return q_fastell, gam

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
    def _parameter_constraints(theta_E, gamma, q, phi_G, s_scale):
        """
        sets bounds to parameters due to numerical stability
        :param theta_E: Einstein radius
        :param gamma: 3d power-law slope
        :param q: axis ratio
        :param phi_G: orientation angle
        :param s_scale: smoothing scale of the core
        :return: input parameters within numerical bounds
        """
        if theta_E < 0:
            theta_E = 0
        if s_scale < 0.00000001:
            s_scale = 0.00000001
        if gamma < 1.2:
            gamma = 1.2
            theta_E = 0
        if gamma > 2.9:
            gamma = 2.9
            theta_E = 0
        if q < 0.01:
            q = 0.01
            theta_E = 0
        if q > 1:
            q = 1.
            theta_E = 0
        return theta_E, gamma, q, phi_G, s_scale
