__author__ = 'sibirrer'

# this file contains a class to compute the truncated Navaro-Frank-White function (Baltz et al 2009)in mass/kappa space
# the potential therefore is its integral

import numpy as np
import scipy.interpolate as interp
import warnings


class TNFW(object):
    """
    this class contains functions concerning the truncated NFW profile with a truncation function (r_trunc^2)*(r^2+r_trunc^2)

    relation are: R_200 = c * Rs

    """
    param_names = ['Rs', 'theta_Rs', 'r_trunc', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'theta_Rs': 0, 'r_trunc': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'theta_Rs': 10, 'r_trunc': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpol=True, num_interp_X=1000, max_interp_X=10):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        """
        self._interpol = interpol
        self._max_interp_X = max_interp_X
        self._num_interp_X = num_interp_X

    def function(self, x, y, Rs, theta_Rs, r_trunc, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param theta_Rs: deflection at Rs
        :param center_x: center of halo
        :param center_y: center of halo
        :return:
        """
        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_ = self.nfwPot(R, Rs, rho0_input, r_trunc)
        #TODO: truncated NFW potential not in place yet!
        return f_

    def L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """

        return np.log(x * (tau + np.sqrt(tau ** 2 + x ** 2)) ** -1)

    def F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """
        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def derivatives(self, x, y, Rs=None, theta_Rs=None, r_trunc=None, center_x=0, center_y=0):

        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, r_trunc, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, theta_Rs, r_trunc, center_x=0, center_y=0):

        #raise Exception('Hessian for truncated nfw profile not yet implemented.')

        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)

        kappa = self.density_2d(x_, y_, Rs, rho0_input, r_trunc)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, r_trunc, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def density(self, R, Rs, rho0, t):
        """
        three dimenstional truncated NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """
        return (t ** 2 * (t ** 2 + R ** 2) ** -1) * rho0 / (R / Rs * (1 + R / Rs) ** 2)

    def density_2d(self, x, y, Rs, rho0, r_trunc, center_x=0, center_y=0):
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
        x = R * Rs ** -1
        tau = float(r_trunc) * Rs ** -1
        Fx = self._F(x, tau)
        return 2 * rho0 * Rs * Fx

    def mass_3d_infinity(self, R, Rs, rho0, t):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        Rs = float(Rs)
        tau = t * Rs ** -1
        m_3d = 4. * np.pi * rho0 * Rs ** 3 * \
               ((tau ** 2 - 1) * np.log(tau) + tau * np.pi - (tau ** 2 + 1))

        return m_3d

    def mass_3d_lens(self, R, Rs, theta_Rs, t):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        rho0 = self._alpha2rho0(theta_Rs, Rs)
        m_3d = self.mass_3d(R, Rs, rho0, t)
        return m_3d

    def nfwPot(self, R, Rs, rho0, r_trunc):
        """
        lensing potential of NFW profile (*Sigma_crit*D_OL**2)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: Epsilon(R) projected density at radius R
        """
        x = R / Rs
        tau = float(r_trunc) / Rs
        hx = self._h(x, tau)
        return 2 * rho0 * Rs ** 3 * hx

    def nfwAlpha(self, R, Rs, rho0, r_trunc, ax_x, ax_y):
        """
        deflection angel of NFW profile (*Sigma_crit*D_OL) along the projection to coordinate 'axis'

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

        x = R / Rs

        tau = float(r_trunc) * Rs ** -1

        gx = self._g(x, tau)
        Fx = self._F(x, tau)

        a = 2*rho0*Rs*(2*gx/x**2 - Fx)  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x

        return a * (ax_y ** 2 - ax_x ** 2) / R ** 2, -a * 2 * (ax_x * ax_y) / R ** 2

    def mass_2d(self,R,Rs,rho0,r_trunc):

        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """

        x = R / Rs
        tau = r_trunc / Rs
        gx = self._g(x,tau)
        m_2d = 4 * rho0 * Rs * R ** 2 * gx / x ** 2 * np.pi
        return m_2d

    def _F(self, X, tau):
        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """
        t2 = tau ** 2
        #Fx = self.F(X)

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
            raise ValueError("The variable type is not compatible with the function, please use float, int or ndarray's.")

        c = 2*_F
        d = -np.pi*(t2+X**2)**-0.5
        e = (t2-1)*(tau*(t2+X**2)**0.5)**-1*self.L(X,tau)
        result = a * (b + c + d + e)

        return result

    def _g(self, x, tau):
        """
        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        return tau ** 2 * (tau ** 2 + 1) ** -2 * (
                (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self.F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
                np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self.L(x, tau) * (tau ** 2 - 1) * tau ** -1))


    def _h(self, X, tau):

        """
        a horrible expression for the integral to compute potential

        :param x: R/Rs
        :param tau: t/Rs
        :type x: float >0
        """

        def cos_func(y):
            if isinstance(y, float) or isinstance(y, int):
                if y > 1:
                    return np.arccosh(y)
                else:
                    return np.arccos(y)
            else:
                values = np.ones_like(y)
                inds1 = np.where(y < 1)
                inds2 = np.where(y > 1)
                values[inds1] = np.arccos(y[inds1])
                # values[inds2] = np.arccosh(y[inds2])
                values[inds2] = np.arccos((1 - y[inds2]) ** .5)
                return values

        def nfw_func(x):
            if isinstance(x,float) or isinstance(x,int):
                if x<1:
                    return np.log(x / 2.) ** 2 - np.arccosh(1. / x) ** 2
                else:
                    return np.log(x / 2.)**2 - np.arccos(1. /x)**2
            else:
                inds1 = np.where(x < 1)
                inds2 = np.where(x >= 1)
                y = np.zeros_like(x)
                y[inds1] = np.log(x[inds1] / 2.) ** 2 - np.arccosh(1. / x[inds1]) ** 2
                y[inds2] = np.log(x[inds2] / 2.) ** 2 - np.arccosh(1. / x[inds2]) ** 2

                return y

        def tnfw_func(x,tau):

            u = x**2
            t2 = tau**2
            Lx = self.L(u**.5, tau)

            return (t2 + 1) ** -2 * (
                2 * t2 * np.pi * (tau - (t2 + u) ** .5 + tau * np.log(tau + (t2 + u) ** .5))
                +
                2 * (t2 - 1) * tau * (t2 + u) ** .5 * Lx
                +
                t2 * (t2 - 1) * Lx ** 2
                +
                4 * t2 * (u - 1) * self.F(u**.5)
                +
                t2 * (t2 - 1) * (cos_func(u ** -0.5)) ** 2
                +
                t2 * ((t2 - 1) * np.log(tau) - t2 - 1) * np.log(u)
                -
                t2 * (
                (t2 - 1) * np.log(tau) * np.log(4 * tau) + 2 * np.log(0.5 * tau) - 2 * tau * (tau - np.pi) * np.log(
                    tau * 2)))

        c = 1e-9
        rescale = 1

        if np.any(X-c<1):

            warnings.warn('Truncated NFW potential not yet implemented for x<1. Using the expression for the NFW '
                          'potential in this regime isntead.')
            rescale = tnfw_func(1.00001,tau)*nfw_func(1)**-1

        if isinstance(X,float) or isinstance(X,int):
            if X<1:
                X = min(X,0.001)
                return nfw_func(X)*rescale
            else:
                return tnfw_func(X,tau)
        else:

            X[np.where(X<0.001)] = 0.001

            values = np.zeros_like(X)
            inds1,inds2 = np.where(X<1),np.where(X>=1)
            values[inds1] = nfw_func(X[inds1])*rescale
            values[inds2] = tnfw_func(X[inds2],tau)

            return values

    def _alpha2rho0(self, theta_Rs, Rs):
        """
        convert angle at Rs into rho0; neglects the truncation
        """
        rho0 = theta_Rs / (4. * Rs ** 2 * (1. + np.log(1. / 2.)))
        return rho0

    def _rho02alpha(self, rho0, Rs):
        """
        neglects the truncation

        convert rho0 to angle at Rs
        :param rho0:
        :param Rs:
        :return:
        """
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return theta_Rs
