__author__ = 'dgilman'

import numpy as np
from lenstronomy.LensModel.Profiles.nfw import NFW
import warnings


class CNFW(object):
    """
    this class computes the lensing quantities of a cored NFW profile:
    rho = rho0 * (r + r_core)^-1 * (r + rs)^-2

    """
    param_names = ['Rs', 'alpha_Rs', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'r_core': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'r_core': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        """
        self._nfw = NFW()

    def function(self, x, y, Rs, alpha_Rs, r_core, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        warnings.warn('Potential for cored NFW potential not yet implemented. '
                      'Using the expression for the NFW '
                      'potential instead.')

        pot = self._nfw.function(x, y, Rs, alpha_Rs, center_x=center_x, center_y=center_y)

        return pot

    def _nfw_func(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """

        c = 0.000001

        if isinstance(x, np.ndarray):
            x[np.where(x<c)] = c
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)

            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)

            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            x = max(x, c)
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def _F(self, X, b, c = 0.001):
        """
        analytic solution of the projection integral

        :param x: a dimensionless quantity, either r/rs or r/rc
        :type x: float >0
        """

        if b == 1:
            b = 1 + c

        prefac = (b - 1) ** -2

        if isinstance(X, np.ndarray):

            X[np.where(X == 1)] = 1 - c

            output = np.empty_like(X)

            inds1 = np.where(np.absolute(X - b)<c)
            output[inds1] = prefac*(-2 - b + (1 + b + b ** 2) * self._nfw_func(b)) * (1 + b) ** -1

            inds2 = np.where(np.absolute(X - b)>=c)

            output[inds2] = prefac * ((X[inds2] ** 2 - 1) ** -1 * (1 - b -
                                    (1 - b * X[inds2] ** 2) * self._nfw_func(X[inds2])) - \
                                        self._nfw_func(X[inds2] * b ** -1))

        else:

            if X == 1:
                X = 1-c

            if np.absolute(X - b)<c:
                output = prefac * (-2 - b + (1 + b + b ** 2) * self._nfw_func(b)) * (1 + b) ** -1

            else:
                output = prefac * ((X ** 2 - 1) ** -1 * (1 - b -
                                    (1 - b * X ** 2) * self._nfw_func(X)) - \
                                        self._nfw_func(X * b ** -1))

        return output

    def _G(self, X, b, c=0.00000001):
        """

        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        if b == 1:
            b = 1+c

        b2 = b ** 2
        x2 = X**2

        fac = (1 - b) ** 2
        prefac = fac ** -1

        if isinstance(X, np.ndarray):

            output = np.ones_like(X)

            inds1 = np.where(np.absolute(X - b) <= c)
            inds2 = np.where(np.absolute(X - b) > c)

            output[inds1] = prefac * (2*(1-2*b+b**3)*self._nfw_func(b) + fac * (-1.38692 + np.log(b2)) - b2*np.log(b2))

            output[inds2] = prefac * (fac * np.log(0.25 * x2[inds2]) - b2 * np.log(b2) + \
                2 * (b2 - x2[inds2]) * self._nfw_func(X[inds2] * b**-1) + 2 * (1+b*(x2[inds2] - 2))*
                             self._nfw_func(X[inds2]))
            return 0.5*output

        else:

            if np.absolute(X - b) <= c:
                output = prefac * (2*(1-2*b+b**3)*self._nfw_func(b) + \
                            fac * (-1.38692 + np.log(b2)) - b2*np.log(b2))
            else:
                output = prefac * (fac * np.log(0.25 * x2) - b2 * np.log(b2) + \
                2 * (b2 - x2) * self._nfw_func(X * b**-1) + 2 * (1+b*(x2 - 2))*
                             self._nfw_func(X))

            return 0.5 * output

    def derivatives(self, x, y, Rs, alpha_Rs, r_core, center_x=0, center_y=0):

        rho0_input = self._alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs, r_core=r_core)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_x, f_y = self.cnfwAlpha(R, Rs, rho0_input, r_core, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, r_core, center_x=0, center_y=0):

        #raise Exception('Hessian for truncated nfw profile not yet implemented.')

        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        rho0_input = self._alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs, r_core=r_core)
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)

        kappa = self.density_2d(x_, y_, Rs, rho0_input, r_core)
        gamma1, gamma2 = self.cnfwGamma(R, Rs, rho0_input, r_core, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def density(self, R, Rs, rho0, r_core):
        """
        three dimenstional truncated NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (central core density)
        :type rho0: float
        :return: rho(R) density
        """

        M0 = 4*np.pi*rho0 * Rs ** 3
        return (M0/4/np.pi) * ((r_core + R)*(R + Rs)**2) ** -1

    def density_2d(self, x, y, Rs, rho0, r_core, center_x=0, center_y=0):
        """
        projected two dimenstional NFW profile (kappa*Sigma_crit)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        b = r_core * Rs ** -1
        x = R * Rs ** -1
        Fx = self._F(x, b)

        return 2 * rho0 * Rs * Fx

    def mass_3d(self, R, Rs, rho0, r_core):
        """
        mass enclosed a 3d sphere or radius r

        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        b = r_core * Rs ** -1
        x = R * Rs ** -1

        M_0 = 4 * np.pi * Rs**3 * rho0

        return M_0 * (x * (1+x) ** -1 * (-1+b) ** -1 + (-1+b) ** -2 *
                      ((2*b-1)*np.log(1/(1+x)) + b **2 * np.log(x / b + 1)))

    def cnfwAlpha(self, R, Rs, rho0, r_core, ax_x, ax_y):
        """
        deflection angel of NFW profile along the projection to coordinate axis

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.00001)
        else:
            R[R <= 0.00001] = 0.00001

        x = R / Rs
        b = r_core * Rs ** -1
        b = max(b, 0.000001)
        gx = self._G(x, b)

        a = 4*rho0*Rs*gx/x**2
        return a * ax_x, a * ax_y

    def cnfwGamma(self, R, Rs, rho0, r_core, ax_x, ax_y):
        """

        shear gamma of NFW profile (times Sigma_crit) along the projection to coordinate 'axis'

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return: Epsilon(R) projected density at radius R
        """
        c = 0.000001
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, c)
        else:
            R[R <= c] = c
        x = R * Rs ** -1
        b = r_core * Rs ** -1
        b = max(b, c)
        gx = self._G(x, b)
        Fx = self._F(x, b)
        a = 2 * rho0 * Rs * (2 * gx / x ** 2 - Fx)  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a * (ax_y ** 2 - ax_x ** 2) / R ** 2, -a * 2 * (ax_x * ax_y) / R ** 2

    def mass_2d(self, R, Rs, rho0, r_core):

        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """

        x = R / Rs
        b = r_core / Rs
        b = max(b, 0.000001)
        gx = self._G(x, b)

        #m_2d = 4 * np.pi* rho0 * Rs**3 * gx

        m_2d = 4*np.pi*rho0*Rs*R**2*gx/x**2

        return m_2d

    def _alpha2rho0(self, alpha_Rs, Rs, r_core):

        b = r_core * Rs ** -1

        gx = self._G(1., b)

        rho0 = alpha_Rs * (4 * Rs ** 2 * gx) ** -1

        return rho0

    def _rho2alpha(self, rho0, Rs, r_core):

        b = r_core * Rs ** -1
        gx = self._G(1., b)
        alpha = 4*Rs ** 2*gx*rho0
        return alpha
