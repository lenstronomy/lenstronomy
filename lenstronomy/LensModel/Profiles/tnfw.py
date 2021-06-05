__author__ = 'sibirrer'

# this file contains a class to compute the truncated Navaro-Frank-White function (Baltz et al 2009)in mass/kappa space
# the potential therefore is its integral

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.nfw import NFW

__all__ = ['TNFW']


class TNFW(LensProfileBase):
    """
    this class contains functions concerning the truncated NFW profile with a truncation function (r_trunc^2)*(r^2+r_trunc^2)

    relation are: R_200 = c * Rs

    """
    profile_name = 'TNFW'
    param_names = ['Rs', 'alpha_Rs', 'r_trunc', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'r_trunc': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'r_trunc': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        """

        """
        self._s = 0.001
        super(LensProfileBase, self).__init__()

    def function(self, x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param alpha_Rs: deflection at Rs
        :param r_trunc: truncation radius
        :param center_x: center of halo
        :param center_y: center of halo
        :return: lensing potential
        """
        rho0_input = self.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        R = np.maximum(R, self._s * Rs)
        f_ = self.nfwPot(R, Rs, rho0_input, r_trunc)

        return f_

    def _L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """
        x = np.maximum(x, self._s)
        return np.log(x * (tau + np.sqrt(tau ** 2 + x ** 2)) ** -1)

    def F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """

        if isinstance(x, np.ndarray):
            x = np.maximum(x, self._s)
            nfwvals = np.zeros_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            inds3 = np.where(x == 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            nfwvals[inds3] = 1
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            x = np.maximum(x, 0)
            if x == 1:
                return 1
            elif x == 0:
                return 0
            elif x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def derivatives(self, x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):

        """
        returns df/dx and df/dy of the function (integral of TNFW), which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_trunc: truncation radius (angular units)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_input = self.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        R = np.maximum(R, self._s * Rs)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, r_trunc, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, r_trunc, center_x=0, center_y=0):

        """
        returns d^2f/dx^2, d^2f/dxdy, d^2f/dydx, d^2f/dy^2 of the TNFW potential f

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_trunc: truncation radius (angular units)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """

        rho0_input = self.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        R = np.maximum(R, self._s * Rs)

        kappa = self.density_2d(x_, y_, Rs, rho0_input, r_trunc)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, r_trunc, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def density(r, Rs, rho0, r_trunc):
        """
        three dimensional truncated NFW profile

        :param r: radius of interest
        :type r: float/numpy array
        :param Rs: scale radius
        :type Rs: float > 0
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: rho(r) density
        """
        return (r_trunc ** 2 * (r_trunc ** 2 + r ** 2) ** -1) * rho0 / (r / Rs * (1 + r / Rs) ** 2)

    def density_2d(self, x, y, Rs, rho0, r_trunc, center_x=0, center_y=0):
        """
        projected two dimensional NFW profile (kappa*Sigma_crit)

        :param R: projected radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        x = R * Rs ** -1
        tau = float(r_trunc) * Rs ** -1
        Fx = self._F(x, tau)
        return 2 * rho0 * Rs * Fx

    def mass_3d(self, r, Rs, rho0, r_trunc):
        """
        mass enclosed a 3d sphere or radius r

        :param r: 3d radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :param r_trunc: truncation radius (angular units)
        :return: M(<r)
        """

        x = r * Rs ** -1
        x = np.maximum(x, self._s)
        func = (r_trunc ** 2 * (-2 * x * (1 + r_trunc ** 2) + 4 * (1 + x) * r_trunc * np.arctan(x / r_trunc) -
                                2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs) + 2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs * (1 + x)) +
                                2 * (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs * r_trunc) -
                                (1 + x) * (-1 + r_trunc ** 2) * np.log(Rs ** 2 * (x ** 2 + r_trunc ** 2)))) / (2. * (1 + x) * (1 + r_trunc ** 2) ** 2)

        m_3d = 4*np.pi*Rs ** 3 * rho0 * func
        return m_3d

    def nfwPot(self, R, Rs, rho0, r_trunc):
        """
        lensing potential of truncated NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :return: lensing potential
        """
        x = R / Rs
        x = np.maximum(x, self._s)
        tau = float(r_trunc) / Rs
        hx = self._h(x, tau)
        return 2 * rho0 * Rs ** 3 * hx

    def nfwAlpha(self, R, Rs, rho0, r_trunc, ax_x, ax_y):
        """
        deflection angel of NFW profile along the projection to coordinate axis

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return:
        """
        R = np.maximum(R, self._s * Rs)
        x = R / Rs
        x = np.maximum(x, self._s)
        tau = float(r_trunc) / Rs
        gx = self._g(x, tau)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    def nfwGamma(self, R, Rs, rho0, r_trunc, ax_x, ax_y):
        """

        shear gamma of NFW profile (times Sigma_crit) along the projection to coordinate 'axis'

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r_trunc: truncation radius (angular units)
        :type r_trunc: float > 0
        :param axis: projection to either x- or y-axis
        :type axis: same as R
        :return:
        """
        R = np.maximum(R, self._s * Rs)
        x = R / Rs
        tau = float(r_trunc) * Rs ** -1
        gx = self._g(x, tau)
        Fx = self._F(x, tau)
        a = 2*rho0*Rs*(2*gx/x**2 - Fx)
        return a * (ax_y ** 2 - ax_x ** 2) / R ** 2, -a * 2 * (ax_x * ax_y) / R ** 2

    def mass_2d(self, R, Rs, rho0, r_trunc):
        """
        analytic solution of the projection integral (convergence)

        :param R: projected radius
        :param Rs: scale radius
        :param rho0: density normalization (characteristic density)
        :param r_trunc: truncation radius (angular units)
        :return: mass enclosed 2d projected cylinder
        """

        x = R / Rs
        x = np.maximum(x, self._s)
        tau = r_trunc / Rs
        gx = self._g(x,tau)
        m_2d = 4 * rho0 * Rs * R ** 2 * gx / x ** 2 * np.pi
        return m_2d

    def _F(self, X, tau):
        """
        analytic solution of the projection integral
        (convergence)

        :param X: R/Rs
        :type X: float >0
        """
        t2 = tau ** 2
        X = np.maximum(X, self._s )
        _F = self.F(X)
        a = t2*(t2+1)**-2
        if isinstance(X, np.ndarray):
            #b = (t2 + 1) * (X ** 2 - 1) ** -1 * (1 - _F)
            b = np.ones_like(X)
            b[X == 1] = (t2+1) * 1./3
            b[X != 1] = (t2 + 1) * (X[X != 1] ** 2 - 1) ** -1 * (1 - _F[X != 1])

        elif isinstance(X, float) or isinstance(X, int):
            if X == 1:
                b = (t2+1)* 1./3
            else:
                b = (t2+1)*(X**2-1)**-1*(1-_F)
        else:
            raise ValueError("The variable type is not compatible with the function, please use float,"
                             " int or ndarray's.")

        c = 2*_F
        d = -np.pi*(t2+X**2)**-0.5
        e = (t2-1)*(tau*(t2+X**2)**0.5)**-1*self._L(X, tau)
        result = a * (b + c + d + e)

        return result

    def _g(self, x, tau):
        """
        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        x = np.maximum(x, self._s)
        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
                (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self.F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
                np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self._L(x, tau) * (tau ** 2 - 1) * tau ** -1))

    @staticmethod
    def _cos_function(x):

        if isinstance(x, np.ndarray) or isinstance(x, list):
            out = np.empty_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x >= 1)

            out[inds1] = -np.arccosh(1 / x[inds1]) ** 2
            out[inds2] = np.arccos(1 / x[inds2]) ** 2

        elif isinstance(x, float) or isinstance(x, int):
            if x < 1:
                out = -np.arccosh(1 / x) ** 2
            else:
                out = np.arccos(1 / x) ** 2

        else:
            raise Exception('x data type %s not recognized.' % x)

        return out

    def _h(self, x, tau):

        """
        expression for the integral to compute potential

        :param x: R/Rs
        :param tau: r_trunc/Rs
        :type x: float >0
        """
        x = np.maximum(x, self._s)

        u = x ** 2
        t2 = tau ** 2
        Lx = self._L(x, tau)
        Fx = self.F(x)

        return (t2 + 1) ** -2 * (
                2 * t2 * np.pi * (tau - (t2 + u) ** .5 + tau * np.log(tau + (t2 + u) ** .5))
                +
                2 * (t2 - 1) * tau * (t2 + u) ** .5 * Lx
                +
                t2 * (t2 - 1) * Lx ** 2
                +
                4 * t2 * (u - 1) * Fx
                +
                t2 * (t2 - 1) * self._cos_function(x)
                +
                t2 * ((t2 - 1) * np.log(tau) - t2 - 1) * np.log(u)
                -
                t2 * (
                        (t2 - 1) * np.log(tau) * np.log(4 * tau) + 2 * np.log(0.5 * tau) - 2 * tau * (
                            tau - np.pi) * np.log(
                    tau * 2)))

    @staticmethod
    def alpha2rho0(alpha_Rs, Rs):
        """
        convert angle at Rs into rho0; neglects the truncation

        :param alpha_Rs: deflection angle at RS
        :param Rs: scale radius
        :return: density normalization (characteristic density)
        """
        return NFW.alpha2rho0(alpha_Rs, Rs)

    @staticmethod
    def rho02alpha(rho0, Rs):
        """
        convert rho0 to angle at Rs; neglects the truncation

        :param rho0: density normalization (characteristic density)
        :param Rs: scale radius
        :return: deflection angle at RS
        """
        return NFW.rho02alpha(rho0, Rs)
