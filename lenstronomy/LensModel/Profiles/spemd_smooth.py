__author__ = 'sibirrer'

import numpy as np



class SPEMD_SMOOTH(object):
    """
    class for smooth power law ellipse mass density profile
    The Einstein ring parameter converts to the definition used by GRAVLENS as follow:
    (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]
    """
    def __init__(self):
        try:
            from fastell4py import fastell4py
            self._fastell4py_bool = True
            self.fastell4py = fastell4py
        except:
            self._fastell4py_bool = False
            print("module fastell4py not installed. You can get it from here: https://github.com/sibirrer/fastell4py")

    def _parameter_constraints(self, theta_E, gamma, q, phi_G, s_scale):
        """
        sets bounds to parameters due to numerical stability
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :param s_scale:
        :return:
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

    def function(self, x, y, theta_E, gamma, q, phi_G, s_scale, center_x=0, center_y=0):
        theta_E, gamma, q, phi_G, s_scale = self._parameter_constraints(theta_E, gamma, q, phi_G, s_scale)
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(theta_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift
        if self._fastell4py_bool:
            potential = self.fastell4py.ellipphi(x1, x2, q_fastell, gam, arat=q, s2=s_scale)
        else:
            return np.zeros_like(x1)
        n = len(np.atleast_1d(x))
        if n <= 1:
            if np.shape(x) == ():
                return np.array(potential[0])
        return potential

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, s_scale, center_x=0, center_y=0):
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
        if self._fastell4py_bool:
            f_x_prim, f_y_prim = self.fastell4py.fastelldefl(x1, x2, q_fastell, gam, arat=q, s2=s_scale)
        else:
            return np.zeros_like(x1), np.zeros_like(x1)
        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim
        n = len(np.atleast_1d(x))
        #if n <= 1:
        #    if np.shape(x) == ():
        #        f_x, f_y = np.array(f_x[0]), np.array(f_y[0])
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, q, phi_G, s_scale, center_x=0, center_y=0):
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
        if self._fastell4py_bool:
            f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = self.fastell4py.fastellmag(x1, x2, q_fastell, gam, arat=q, s2=s_scale)
        else:
            return np.zeros_like(x1), np.zeros_like(x1), np.zeros_like(x1)
        n = len(np.atleast_1d(x))
        if n <= 1:
            if np.shape(x) == ():
                f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = np.array(f_x_prim[0]), np.array(f_y_prim[0]), np.array(f_xx_prim[0]), np.array(f_yy_prim[0]), np.array(f_xy_prim[0])
        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def convert_params(self, theta_E, gamma, q):
        """

        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :return:   prefactor to SPEMP profile for FASTELL
        """
        gam = (gamma-1)/2.
        q_fastell = (3-gamma)/2. * (theta_E ** 2 / q) ** gam

        return q_fastell, gam