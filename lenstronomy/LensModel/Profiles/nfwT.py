__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
# the potential therefore is its integral

import numpy as np


class NFWt(object):
    """
    this class contains functions concerning the truncated NFW profile with truncation function (t^2 / (r^2 + t^2))
    detailed in Baltz et al 2008

    relation are: R_200 = c * Rs
    """

    def function(self, x, y, Rs, theta_Rs, t, center_x=0, center_y=0):
        """

        :param x: angular position
        :param y: angular position
        :param Rs: angular turn over point
        :param theta_Rs: deflection at Rs
        :param t: truncation radius (angular units)
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
        f_ = self.nfwPot(R, Rs, rho0_input, t)
        return f_

    def derivatives(self, x, y, Rs, theta_Rs, t, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, t, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, theta_Rs, t, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        kappa = self.density_2d(R, 0, Rs, rho0_input, t)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, t, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def density(self, R, Rs, rho0, t):
        """
        three dimenstional NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param t: truncation radius (angular units)
        :return: rho(R) density
        """
        return rho0 / (R / Rs * (1 + R / Rs) ** 2) * (t**2*(R**2 + t**2)**-1)

    def density_2d(self, x, y, Rs, rho0, t, center_x=0, center_y=0):
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
        :param t: truncation radius (angular units)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        x = R / Rs
        Fx = self._F(x,t)
        return 2 * rho0 * Rs * Fx

    def mass_3d(self, R, Rs, rho0, t):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :param t: truncation radius (angular units)
        :return:
        """
        raise ValueError('not yet implemented')

    def mass_3d_lens(self, R, Rs, theta_Rs, t):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :param t: truncation radius (angular units)
        :return:
        """
        raise ValueError('not yet implemented')

    def mass_2d(self, R, Rs, rho0, t):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :param t: truncation radius (angular units)
        :return:
        """
        raise ValueError('not yet implemented')

    def nfw2D_smoothed(self, R, Rs, rho0, t, pixscale):
        """
        projected two dimenstional NFW profile with smoothing around the pixel scale
        this routine is ment to better compare outputs to N-body simulations (not ment ot do lensemodelling with it)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param pixscale: pixel scale (same units as R,Rs)
        :type pixscale: float>0
        :param t: truncation radius (angular units)
        :return: Epsilon(R) projected density at radius R
        """
        x = R / Rs
        d = pixscale / (2 * Rs)
        a = np.empty_like(x)
        x_ = x[x > d]
        upper = x_ + d
        lower = x_ - d

        a[x > d] = 4 * rho0 * Rs ** 3 * (self._g(upper,t) - self._g(lower,t)) / (2 * x_ * Rs * pixscale)
        a[x < d] = 4 * rho0 * Rs ** 3 * self._g(d,t) / ((pixscale / 2) ** 2)
        return a

    def nfwPot(self, R, Rs, rho0, t):
        """
        lensing potential of NFW profile (*Sigma_crit*D_OL**2)

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param r200: radius of (sub)halo
        :type r200: float>0
        :param t: truncation radius (angular units)
        :return: Epsilon(R) projected density at radius R
        """
        x = R / Rs
        hx = self._h(x, t)
        return 2 * rho0 * Rs ** 3 * hx

    def nfwAlpha(self, R, Rs, rho0, t, ax_x, ax_y):
        """
        deflection angel of NFW profile (*Sigma_crit*D_OL) along the projection to coordinate "axis"

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
        :param t: truncation radius (angular units)
        :return: Epsilon(R) projected density at radius R
        """
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, 0.00001)
        else:
            R[R <= 0.00001] = 0.00001
        x = R / Rs
        gx = self._g(x,t)
        a = 4 * rho0 * Rs * R * gx / x ** 2 / R
        return a * ax_x, a * ax_y

    def nfwGamma(self, R, Rs, rho0, t, ax_x, ax_y):
        """
        shear gamma of NFW profile (*Sigma_crit) along the projection to coordinate "axis"

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
        c = 0.001
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, c)
        else:
            R[R <= c] = c

        x = R/Rs
        gx = self._g(x,t)
        Fx = self._F(x,t)
        a = 2*rho0*Rs*(2*gx/x**2 - Fx)#/x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a*(ax_y**2-ax_x**2)/R**2, -a*2*(ax_x*ax_y)/R**2

    def _F(self, X,t):
        """
        analytic solution of the projection integral

        :param x: R/Rs
        :type x: float >0
        """
        c = 0.001
        t2 = t**2
        t2p = (t2+1)**2
        t2m = (t2-1)**2

        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(c, X)
                cos = np.arccosh(x ** -1)
                F = cos * (1 - x ** 2) ** -.5

            elif X == 1:
                cos = 0
                F = 1

            else:  # X > 1:
                cos = np.arccos(X ** -1)
                F = cos * (X ** 2 - 1) ** -.5

            a = t2*t2p**-1*(t2p*(X**2-1)**-1 * (1-F) +2*F - np.pi*(t2+X**2)**-.5 +
                            t2m * self._Log(X,t)*(t*(t2+X**2)**.5)**-1)

        else:

            a = np.empty_like(X)
            X[X < c] = c

            x = X[X < 1]
            cos = np.arccosh(x ** -1)
            F = cos * (1 - x ** 2) ** -.5
            a[X < 1] = t2*t2p**-1*(t2p*(x**2-1)**-1 * (1-F) +2*F - np.pi*(t2+x**2)**-.5 +
                            t2m * self._Log(x,t)*(t*(t2+x**2)**.5)**-1)

            x = X[X == 1]
            cos = 0
            F = 1
            a[X == 1] = t2*t2p**-1*(t2p*(x**2-1)**-1 * (1-F) +2*F - np.pi*(t2+x**2)**-.5 +
                            t2m * self._Log(x,t)*(t*(t2+x**2)**.5)**-1)

            x = X[X > 1]
            cos = np.arccos(x ** -1)
            F = cos * (x ** 2 - 1) ** -.5
            a[X > 1] = t2*t2p**-1*(t2p*(x**2-1)**-1 * (1-F) +2*F - np.pi*(t2+x**2)**-.5 +
                            t2m * self._Log(x,t)*(t*(t2+x**2)**.5)**-1)

        return a


    def _g(self, X, t):

        c = 0.001

        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(c, X)
                cos = np.arccosh(x**-1)
                F = cos*(1 - x ** 2) ** -.5

            elif X == 1:
                cos = 0
                F = 1

            else:  # X > 1:
                cos = np.arccos(X ** -1)
                F = cos * (X ** 2 - 1) ** -.5

            a = t ** 2 * (t ** 2 + 1) ** -2 * (
            (t ** 2 + 1 + 2 * (X ** 2 - 1)) * F + t * np.pi + (t ** 2 - 1) * np.log(t) +
            np.sqrt(t ** 2 + X ** 2) * (-np.pi + self._Log(X, t) * (t ** 2 - 1) * t ** -1))


        else:

            a = np.empty_like(X)
            X[X < c] = c

            x = X[X < 1]
            cos = np.arccosh(x ** -1)
            F = cos * (1 - x ** 2) ** -.5
            a[X < 1] = t ** 2 * (t ** 2 + 1) ** -2 * (
            (t ** 2 + 1 + 2 * (X ** 2 - 1)) * F + t * np.pi + (t ** 2 - 1) * np.log(t) +
            np.sqrt(t ** 2 + X ** 2) * (-np.pi + self._Log(X, t) * (t ** 2 - 1) * t ** -1))

            x = X[X == 1]
            cos = 0
            F = 1
            a[X == 1] = t ** 2 * (t ** 2 + 1) ** -2 * (
            (t ** 2 + 1 + 2 * (X ** 2 - 1)) * F + t * np.pi + (t ** 2 - 1) * np.log(t) +
            np.sqrt(t ** 2 + X ** 2) * (-np.pi + self._Log(X, t) * (t ** 2 - 1) * t ** -1))

            x = X[X > 1]
            cos = np.arccos(x ** -1)
            F = cos * (x ** 2 - 1) ** -.5
            a[X > 1] = t ** 2 * (t ** 2 + 1) ** -2 * (
            (t ** 2 + 1 + 2 * (X ** 2 - 1)) * F + t * np.pi + (t ** 2 - 1) * np.log(t) +
            np.sqrt(t ** 2 + X ** 2) * (-np.pi + self._Log(X, t) * (t ** 2 - 1) * t ** -1))

        return a


    def _Log(self, x, t):
        return np.log(x*(t+np.sqrt(t**2+x**2))**-1)

    def _h(self, X, t):
        """
        analytic solution of integral for NFW profile to compute the potential

        :param x: R/Rs
        :type x: float >0
        """
        c = 0.001

        if isinstance(X, int) or isinstance(X, float):

            x = max(0.001, X)
            t2 = t ** 2
            u = x ** 2
            t2m = (t2 - 1) ** 2
            t2p = (t2 + 1) ** 2
            t2u = np.sqrt(t2 + u)
            L = self._Log(u ** .5, t)

            if X < 1:
                cos = np.arccosh(u ** -.5)
                F = cos * (1 - x**2)**-.5
            elif X>1:  # X >= 1:
                cos = np.arccos(u ** -.5)
                F = cos * (x ** 2 - 1) ** -.5
            else:
                cos = 0
                F = 1

            a = t2p ** -1 * (2 * t2 * np.pi * (t - t2u + t * np.log(t + t2u)) +
                         2 * t2m * t * t2u * L +
                         t2 * t2m * L ** 2 +
                         4 * t2 * (u - 1) * F +
                         t2 * t2m * cos ** 2 +
                         t2 * (t2m * np.log(t) - t2 - 1) * np.log(u) -
                         t2 * (t2m * np.log(t) * np.log(4 * t) + 2 * np.log(0.5 * t) - 2 * t * (t - np.pi) * np.log(
                             2 * t)))
        else:

            X[X <= c] = 0.001
            t2 = t ** 2
            t2m = (t2 - 1) ** 2
            t2p = (t2 + 1) ** 2

            a = np.empty_like(X)

            x = X[X < 1]
            u = x ** 2
            t2u = np.sqrt(t2 + u)
            L = self._Log(u ** .5, t)
            cos = np.arccosh(u ** -.5)
            F = cos * (1 - x ** 2) ** -.5

            a[X < 1] = t2p ** -1 * (2 * t2 * np.pi * (t - t2u + t * np.log(t + t2u)) +
                         2 * t2m * t * t2u * L +
                         t2 * t2m * L ** 2 +
                         4 * t2 * (u - 1) * F +
                         t2 * t2m * cos ** 2 +
                         t2 * (t2m * np.log(t) - t2 - 1) * np.log(u) -
                         t2 * (t2m * np.log(t) * np.log(4 * t) + 2 * np.log(0.5 * t) - 2 * t * (t - np.pi) * np.log(
                             2 * t)))

            x = X[X > 1]
            u = x ** 2
            t2u = np.sqrt(t2 + u)
            L = self._Log(u ** .5, t)
            cos = np.arccos(u ** -.5)
            F = cos * (x ** 2 - 1) ** -.5
            a[X > 1] = t2p ** -1 * (2 * t2 * np.pi * (t - t2u + t * np.log(t + t2u)) +
                         2 * t2m * t * t2u * L +
                         t2 * t2m * L ** 2 +
                         4 * t2 * (u - 1) * F +
                         t2 * t2m * cos ** 2 +
                         t2 * (t2m * np.log(t) - t2 - 1) * np.log(u) -
                         t2 * (t2m * np.log(t) * np.log(4 * t) + 2 * np.log(0.5 * t) - 2 * t * (t - np.pi) * np.log(
                             2 * t)))

            x = X[X == 1]
            u = x ** 2
            t2u = np.sqrt(t2 + u)
            L = self._Log(u ** .5, t)
            cos = 0
            F = 1
            a[X == 1] = t2p ** -1 * (2 * t2 * np.pi * (t - t2u + t * np.log(t + t2u)) +
                                     2 * t2m * t * t2u * L +
                                     t2 * t2m * L ** 2 +
                                     4 * t2 * (u - 1) * F +
                                     t2 * t2m * cos ** 2 +
                                     t2 * (t2m * np.log(t) - t2 - 1) * np.log(u) -
                                     t2 * (t2m * np.log(t) * np.log(4 * t) + 2 * np.log(0.5 * t) - 2 * t * (
                                     t - np.pi) * np.log(
                                         2 * t)))

        return a

    def _alpha2rho0(self, theta_Rs, Rs):
        """
        convert angle at Rs into rho0
        """
        rho0 = theta_Rs / (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return rho0

    def _rho02alpha(self, rho0, Rs):
        """
        convert rho0 to angle at Rs
        :param rho0:
        :param Rs:
        :return:
        """
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return theta_Rs


