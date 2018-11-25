__author__ = 'dgilman'

# this file contains a class to compute the truncated Navaro-Frank-White function (Baltz et al 2009)in mass/kappa space
# the potential therefore is its integral

import numpy as np

class coreBurkert(object):

    """
    lensing properties of a modified Burkert profile with variable core size
    normalized by rho0, the central core density

    """

    param_names = ['Rs', 'rho0', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 1, 'rho0': 1e+5, 'r_core': 0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'rho0': 1e+10, 'r_core': 50, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, Rs, rho0, r_core, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param rho0: core density
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """

        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_ = self.cBurkPot(R, Rs, rho0, r_core)
        return f_

    def derivatives(self, x, y, Rs=None, rho0=None, r_core=None, center_x=0, center_y=0):
        """
        deflection angles
        :param x: x coordinate
        :param y: y coordinate
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        :param center_x:
        :param center_y:
        :return:
        """
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)

        dx, dy = self.coreBurkAlpha(R, Rs, rho0, r_core, x_, y_)

        return dx, dy

    def mass_2d(self, R, Rs, rho0, r_core):

        """
        analytic solution of the projection integral
        (convergence)

        :param R: projected distance
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        """

        x = R / Rs
        p = Rs / r_core
        gx = self._G(x, p)

        m_2d = 2 * np.pi * rho0 * Rs ** 3 * gx

        return m_2d

    def coreBurkAlpha(self, R, Rs, rho0, r_core, ax_x, ax_y):
        """
        deflection angle

        :param R:
        :param Rs:
        :param rho0:
        :param r_core:
        :param ax_x:
        :param ax_y:
        :return:
        """
        x = R / Rs
        p = Rs / r_core

        gx = self._G(x, p)

        a = 2 * rho0 * Rs**2 * gx / x

        return a * ax_x / R, a * ax_y / R

    def hessian(self, x, y, Rs, rho0, r_core, center_x=0, center_y=0):

        """

        :param x: x coordinate
        :param y: y coordinate
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        :param center_x:
        :param center_y:
        :return:
        """

        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)

        kappa = self.density_2d(x_, y_, Rs, rho0, r_core)

        gamma1, gamma2 = self.cBurkGamma(R, Rs, rho0, r_core, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def density(self, R, Rs, rho0, r_core):
        """
        three dimenstional cored Burkert profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: central core density
        :type rho0: float
        :return: rho(R) density
        """

        return rho0 * ((1 + R * r_core ** -1) * (1 + (R*Rs**-1)**2) )**-1

    def density_2d(self, x, y, Rs, rho0, r_core, center_x=0, center_y=0):
        """
        projected two dimenstional core Burkert profile (kappa*Sigma_crit)

        :param x: x coordinate
        :param y: y coordinate
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        x = R * Rs ** -1
        p = Rs * r_core ** -1
        Fx = self._F(x, p)

        return 2 * rho0 * Rs * Fx

    def mass_3d(self, R, Rs, rho0, r_core):
        """
        :param R: projected distance
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        """

        Rs = float(Rs)
        p = Rs * r_core**-1
        c = R / Rs

        factor = 0.5*(1+p**2)**-1 * (p*(2*np.arctan(c**-1) - np.pi) + p**2*np.log(1+c**2)+2*np.log(p*c+1))

        return rho0 * factor * (4*np.pi*Rs**3*p)

    def cBurkPot(self, R, Rs, rho0, r_core):

        """
        :param R: projected distance
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        """
        x = R / Rs
        p = Rs / r_core
        hx = self._H(x, p)

        return 2 * rho0 * Rs ** 3 * hx

    def cBurkGamma(self, R, Rs, rho0, r_core, ax_x, ax_y):
        """

        :param R: projected distance
        :param Rs: scale radius
        :param rho0: central core density
        :param r_core: core radius
        :param ax_x: x coordinate
        :param ax_y: y coordinate
        :return:
        """
        c = 0.000001
        
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, c)
        else:
            R[R <= c] = c

        x = R * Rs**-1
        p = Rs * r_core**-1
        
        gx = self._G(x, p)
        fx = self._F(x, p)

        m_x = 2*rho0*Rs**3 * gx
        kappa = 2*rho0*Rs*fx

        a = 2*(m_x * R**-2 - kappa)

        return 0.5 * a * (ax_y ** 2 - ax_x ** 2) / R**2, -a * (ax_x * ax_y) / R**2

    def _u(self, x):

        return np.sqrt(1 + x**2)

    def _g(self, x, p):

        return np.sqrt(1 - x**2 * p**2)

    def _f(self, x, p):

        return np.sqrt(x ** 2 * p ** 2 - 1)

    def _H(self, x, p):

        prefactor = (p + p**3)**-1

        if isinstance(x, np.ndarray):

            inds0 = np.where(x * p == 1)
            inds1 = np.where(x * p < 1)
            inds2 = np.where(x * p > 1)
            func = np.ones_like(x)

            func[inds1] = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x[inds1]) -
                                                0.9058413472016892 * p) * p + 2 * p ** 2 * (
                                  self._u(x[inds1]) - 0.5 * np.arctanh(self._u(x[inds1])**-1)) * np.arctanh(
                self._u(x[inds1])**-1) + \
                          2 * (self._g(x[inds1], p) - 0.5 * np.arctanh(self._g(x[inds1], p))) * \
                          np.arctanh(self._g(x[inds1], p)) + (1 + p ** 2) * np.log(x[inds1]) ** 2 - np.pi * p * \
                          np.log(1 + self._u(x[inds1])) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                          np.log(p ** 2) + np.log(x[inds1]) * (
                                      0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))

            func[inds2] = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x[inds2]) -
                                                0.9058413472016892 * p) * p + 2 * p ** 2 * (
                                      self._u(x[inds2]) - 0.5 * np.arctanh(self._u(x[inds2])**-1)) * np.arctanh(
                self._u(x[inds2])**-1) + \
                          -2 * (self._f(x[inds2], p) - 0.5 * np.arctan(self._f(x[inds2], p))) * \
                          np.arctan(self._f(x[inds2], p)) + (1 + p ** 2) * np.log(x[inds2]) ** 2 - np.pi * p * \
                          np.log(1 + self._u(x[inds2])) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                          np.log(p ** 2) + np.log(x[inds2]) * (
                                      0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))

            func[inds0] = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x[inds0]) -
                                                0.9058413472016892 * p) * p + 2 * p ** 2 * (
                                  self._u(x[inds0]) - 0.5 * np.arctanh(self._u(x[inds0])**-1)) * np.arctanh(
                self._u(x[inds0])**-1) \
                          + (1 + p ** 2) * np.log(x[inds0]) ** 2 - np.pi * p * \
                          np.log(1 + self._u(x[inds0])) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                          np.log(p ** 2) + np.log(x[inds0]) * (
                                      0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))

        else:
            if x * p < 1:
                func = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x) -
                                             0.9058413472016892 * p) * p + 2 * p ** 2 * (
                               self._u(x) - 0.5 * np.arctanh(self._u(x)**-1)) * np.arctanh(self._u(x)**-1) + \
                       2 * (self._g(x, p) - 0.5 * np.arctanh(self._g(x, p))) * \
                       np.arctanh(self._g(x, p)) + (1 + p ** 2) * np.log(x) ** 2 - np.pi * p * \
                       np.log(1 + self._u(x)) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                       np.log(p ** 2) + np.log(x) * (0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))
            elif x * p > 1:
                func = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x) -
                                                0.9058413472016892 * p) * p + 2 * p ** 2 * (
                                      self._u(x) - 0.5 * np.arctanh(self._u(x)**-1)) * np.arctanh(
                self._u(x)**-1) + \
                          -2 * (self._f(x, p) - 0.5 * np.arctan(self._f(x, p))) * \
                          np.arctan(self._f(x, p)) + (1 + p ** 2) * np.log(x) ** 2 - np.pi * p * \
                          np.log(1 + self._u(x)) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                          np.log(p ** 2) + np.log(x) * (
                                      0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))
            else:
                func = 0.9058413472016891 + (-0.9640065632861909 + np.pi * self._u(x) -
                                             0.9058413472016892 * p) * p + 2 * p ** 2 * (
                               self._u(x) - 0.5 * np.arctanh(self._u(x)**-1)) * np.arctanh(self._u(x)**-1) \
                       + (1 + p ** 2) * np.log(x) ** 2 - np.pi * p * \
                       np.log(1 + self._u(x)) + (0.3068528194400547 + 0.25 * np.log(p ** 2)) * \
                       np.log(p ** 2) + np.log(x) * (0.6137056388801094 + 0.6137056388801094 * p ** 2 + np.log(p ** 2))

        return prefactor * func

    def _F(self, x, p):

        """
        solution of the projection integal (kappa)
        arctanh / arctan function
        :param x: r/Rs
        :param p: r_core / Rs
        :return:
        """
        prefactor = 0.5*(1 + p**2)**-1

        if isinstance(x, np.ndarray):

            inds0 = np.where(x*p == 1)
            inds1 = np.where(x * p < 1)
            inds2 = np.where(x * p > 1)

            func = np.ones_like(x)

            func[inds0] = self._u(x[inds0])**-1 * (np.pi + 2*p*np.arctanh(self._u(x[inds0])**-1))

            func[inds1] = self._u(x[inds1])**-1 * (np.pi + 2*p*np.arctanh(self._u(x[inds1])**-1)) - \
                          (2 * p * self._g(x[inds1], p)**-1 * np.arctanh(self._g(x[inds1], p)))

            func[inds2] = self._u(x[inds2]) ** -1 * (np.pi + 2 * p * np.arctanh(self._u(x[inds2]) ** -1)) - \
                          (2 * p * self._f(x[inds2], p) ** -1 * np.arctan(self._f(x[inds2], p)))

            return prefactor * func

        else:

            if x*p == 1:
                func = self._u(x)**-1 * (np.pi + 2*p*np.arctanh(self._u(x)**-1))
            elif x*p<1:
                func = self._u(x)**-1 * (np.pi + 2*p*np.arctanh(self._u(x)**-1)) - \
                          (2 * p * self._g(x, p)**-1 * np.arctanh(self._g(x, p)))
            else:
                func = self._u(x) ** -1 * (np.pi + 2 * p * np.arctanh(self._u(x) ** -1)) - \
                          (2 * p * self._f(x, p) ** -1 * np.arctan(self._f(x, p)))

            return prefactor * func

    def _G(self, x, p):
        """
        analytic solution of the 2d projected mass integral
        integral: 2 * pi * x * kappa * dx
        :param x:
        :param p:
        :return:
        """

        prefactor = (p + p**3)**-1

        if isinstance(x, np.ndarray):

            inds0 = np.where(x*p == 1)
            inds1 = np.where(x*p < 1)
            inds2 = np.where(x*p > 1)

            func = np.ones_like(x)

            func[inds0] = np.log(0.25*x[inds0]**2*p**2) + np.pi * p * (self._u(x[inds0]) - 1) + \
                          2 * p ** 2 * (self._u(x[inds0])*np.arctanh(self._u(x[inds0])**-1) +
                                        np.log(0.5*x[inds0]))

            func[inds1] = np.log(0.25*x[inds1]**2*p**2) + np.pi * p * (self._u(x[inds1]) - 1) + \
                          2 * p ** 2 * (self._u(x[inds1])*np.arctanh(self._u(x[inds1])**-1) +
                                        np.log(0.5*x[inds1])) + 2 * self._g(x[inds1], p) * np.arctanh(self._g(x[inds1], p))

            func[inds2] = np.log(0.25 * x[inds2] ** 2 * p ** 2) + np.pi * p * (self._u(x[inds2]) - 1) + \
                          2 * p ** 2 * (self._u(x[inds2]) * np.arctanh(self._u(x[inds2]) ** -1) +
                                        np.log(0.5 * x[inds2])) - 2 * self._f(x[inds2], p) * np.arctan(self._f(x[inds2], p))


        else:

            if x*p == 1:

                func = np.log(0.25 * x ** 2 * p ** 2) + np.pi * p * (self._u(x) - 1) + \
                          2 * p ** 2 * (self._u(x) * np.arctanh(self._u(x) ** -1) +
                                        np.log(0.5 * x))

            elif x*p < 1:

                func = np.log(0.25 * x ** 2 * p ** 2) + np.pi * p * (self._u(x) - 1) + \
                          2 * p ** 2 * (self._u(x) * np.arctanh(self._u(x) ** -1) +
                                        np.log(0.5 * x)) + 2 * self._g(x, p) * np.arctanh(self._g(x, p))

            else:

                func = np.log(0.25 * x ** 2 * p ** 2) + np.pi * p * (self._u(x) - 1) + \
                       2 * p ** 2 * (self._u(x) * np.arctanh(self._u(x) ** -1) +
                                     np.log(0.5 * x)) + 2 * self._f(x, p) * np.arctan(self._f(x, p))

        return func * prefactor