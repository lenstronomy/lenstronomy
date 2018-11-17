__author__ = 'sibirrer'

# this file contains a class to compute the truncated Navaro-Frank-White function (Baltz et al 2009)in mass/kappa space
# the potential therefore is its integral

import numpy as np
import scipy.interpolate as interp
import warnings


class coreBurkert(object):
    """
    this class contains functions concerning the truncated NFW profile with a truncation function (r_core^2)*(r^2+r_core^2)

    relation are: R_200 = c * Rs

    """
    param_names = ['Rs', 'theta_Rs', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'theta_Rs': 0, 'r_core': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'theta_Rs': 10, 'r_core': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self, Rmax_norm = 10):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        """
        self._Rmax_norm = Rmax_norm

    def function(self, x, y, Rs, theta_Rs, r_core, center_x=0, center_y=0):
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

    def F(self, x, p):
        """
        solution of the projection integal (kappa)
        arctanh / arctan function
        :param x: r/Rs
        :param p: r_core / Rs
        :return:
        """

        if isinstance(x, np.ndarray):

            x2 = x**2
            xp2 = x2*p**2

            inds1 = np.where(x * p < 1)
            inds2 = np.where(x * p > 1)

            func = np.ones_like(x)
            inds0 = np.where(x*p == 1)
            func[inds0] = (1+p**2)**-1*(np.pi * np.sqrt(1+x2[inds0]) ** -1 + 2*p*np.arctanh(np.sqrt(1+x2[inds0]) ** -1)*np.sqrt(1+x2[inds0]) ** -1)

            arg = np.sqrt(1 - xp2[inds1])
            arg1 = np.sqrt(1+x2[inds1]) ** -1
            func[inds1] = (1+p**2)**-1*(np.pi * arg1 + 2*p*np.arctanh(arg1)*arg1 -2*p*np.arctanh(arg) * arg ** -1)

            arg = np.sqrt(xp2[inds2] - 1)
            arg2 = np.sqrt(1+x2[inds2]) ** -1
            func[inds2] = (1+p**2)**-1*(np.pi * arg2 + 2*p*np.arctanh(arg2)*arg2 + p * np.real(1j* np.log((1j - arg) * (1j + arg)**-1)) * arg ** -1)

            return func

        else:

            if x*p == 1:
                return (1+p**2)**-1*(np.pi * np.sqrt(1 + x) ** -1 +
                                     2*p*np.arctanh(np.sqrt(1 + x) ** -1)*np.sqrt(1 + x) ** -1)
            elif x*p<1:
                arg = np.sqrt(1 - x**2*p**2)
                arg1 = np.sqrt(1 + x) ** -1
                return (1+p**2)**-1*(np.pi * arg1 + 2*p*np.arctanh(arg1)*arg1 -2*p*np.arctanh(arg) * arg ** -1)
            else:
                arg = np.sqrt(-1 + x ** 2 * p ** 2)
                arg1 = np.sqrt(1 + x) ** -1
                return (1+p**2)**-1*(np.pi * arg1 + 2*p*np.arctanh(arg1)*arg1 + p * np.real(1j* np.log((1j - arg) * (1j + arg)**-1)) * arg ** -1)

    def G(self, x, p):

        """
        analytic solution of the 2d projected mass integral
        integral: 2 * pi * x * kappa * dx
        :param x:
        :param p:
        :return:
        """
        
        if isinstance(x, np.ndarray):
            func = np.ones_like(x)
            inds1 = np.where(x*p < 1)
    
            func[inds1] = (2*np.pi*(p+p**3)**-1)*(np.pi * (-1 + np.sqrt(1 + x[inds1] ** 2)) *p + p ** 2 * (2 * np.log(x[inds1] / 2.) + np.sqrt(1 + x[inds1] ** 2) * np.log((2 + x[inds1] ** 2 + 2 *
             np.sqrt(1 + x[inds1] ** 2)) / x[inds1] ** 2))+ np.log((x[inds1] ** 2 *p ** 2) / 4.) - np.sqrt(1 - x[inds1] ** 2 *p ** 2) * \
            np.log((2 - x[inds1] ** 2 *p ** 2 - 2 * np.sqrt(1 - x[inds1] ** 2 * p ** 2)) / (x[inds1] ** 2 *p ** 2)))
    
            inds2 = np.where(x*p > 1)
            func[inds2] = (2*np.pi*(p+p**3)**-1)*(np.pi*(-1 + np.sqrt(1 + x[inds2]**2))*p -2*np.sqrt(-1 + x[inds2]**2*p**2)*\
                          np.arctan(np.sqrt(-1 + x[inds2]**2*p**2)) - np.log(4) + p**2*(2*np.log(x[inds2]/2.) + 
                         np.sqrt(1 + x[inds2]**2)*np.log((2 + x[inds2]**2 + 2*np.sqrt(1 + x[inds2]**2))/x[inds2]**2)) + \
                          2*np.log(x[inds2]*p))

            inds0 = np.where(x*p == 1)
            func[inds0] = (2 * np.pi * (p + p ** 3) ** -1) * (
                        np.pi * (-1 + np.sqrt(1 + x[inds0] ** 2)) * p -  np.log(4) + p ** 2 * (
                                    2 * np.log(x[inds0] / 2.) +
                                    np.sqrt(1 + x[inds0] ** 2) * np.log(
                                (2 + x[inds0] ** 2 + 2 * np.sqrt(1 + x[inds0] ** 2)) / x[inds0] ** 2)) + \
                        2 * np.log(x[inds0] * p))

            return func

        
        else:
            if x*p == 1:
                return (2*np.pi*(p+p**3)**-1)*(np.pi*(-1 + np.sqrt(1 + x**2))*p - np.log(4) + p**2*(2*np.log(x/2.) +
                         np.sqrt(1 + x**2)*np.log((2 + x**2 + 2*np.sqrt(1 + x**2))/x**2)) + \
                          2*np.log(x*p))
            elif x*p < 1:
                return (2*np.pi*(p+p**3)**-1)*(np.pi * (-1 + np.sqrt(1 + x ** 2)) *p + p ** 2 * (2 * np.log(x / 2.) + np.sqrt(1 + x ** 2) * np.log((2 + x ** 2 + 2 *
             np.sqrt(1 + x ** 2)) / x ** 2))+ np.log((x ** 2 *p ** 2) / 4.) - np.sqrt(1 - x ** 2 *p ** 2) * \
            np.log((2 - x ** 2 *p ** 2 - 2 * np.sqrt(1 - x ** 2 * p ** 2)) / (x ** 2 *p ** 2)))
            else:
                return (2*np.pi*(p+p**3)**-1)*(np.pi*(-1 + np.sqrt(1 + x**2))*p -2*np.sqrt(-1 + x**2*p**2)*\
                          np.arctan(np.sqrt(-1 + x**2*p**2)) - np.log(4) + p**2*(2*np.log(x/2.) + 
                         np.sqrt(1 + x**2)*np.log((2 + x**2 + 2*np.sqrt(1 + x**2))/x**2)) + \
                          2*np.log(x*p))

    def derivatives(self, x, y, Rs=None, theta_Rs=None, r_core=None, center_x=0, center_y=0):

        rho0 = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)

        rho0 *= self._rescale_density(Rs, r_core)

        dx, dy = self.coreBurkAlpha(R, Rs, rho0, r_core, x_, y_)

        return dx, dy

    def coreBurkAlpha(self, R, Rs, rho0, r_core, ax_x, ax_y):

        x = R / Rs
        p = Rs / r_core

        gx = self.G(x, p)
        a = rho0 * Rs * R * gx / x ** 2 / R
        return a * ax_x, a * ax_y

    def hessian(self, x, y, Rs, theta_Rs, r_core, center_x=0, center_y=0):

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

        kappa = self.density_2d(x_, y_, Rs, rho0_input, r_core)

        gamma1, gamma2 = self.cBurkGamma(R, Rs, self._rescale_density(Rs, r_core)*rho0_input, r_core, x_, y_)
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
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """

        return self._rescale_density(Rs, r_core)*rho0 * ((1 + R * r_core ** -1) * (1 + (R*Rs**-1)**2) )**-1

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
        x = R * Rs ** -1
        p = Rs * r_core ** -1
        Fx = self.F(x, p)

        return rho0 * Rs * Fx * self._rescale_density(Rs, r_core)

    def mass_3d(self, R, Rs, rho0, r_core):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        Rs = float(Rs)
        m_0 = 4*np.pi*Rs**3 * rho0
        p = Rs * r_core**-1
        c = R / Rs

        factor = 0.5*(1+p**2)**-1 * (p*(2*np.arctan(c**-1) - np.pi) + p**2*np.log(1+c**2)+2*np.log(p*c+1))

        return m_0*factor

    def _rescale_density(self, Rs, r_core):
        """
        Rescales cored Burkert quantities such that the mass enclosed (in three-D) within Rmax
        is the same as an NFW profile with scale radius Rs
        :param Rmax:
        :param Rs:
        :param r_core:
        :return:
        """
        Rmax = self._Rmax_norm * Rs
        m3d_coreburk = self.mass_3d(Rmax, Rs, 1, r_core)
        m3d_nfw = 4*np.pi*Rs**3*(np.log((Rs + Rmax)/Rs) - Rmax/(Rs + Rmax))

        return (m3d_nfw * m3d_coreburk**-1)

    def nfwPot(self, R, Rs, rho0, r_core):
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
        p = float(r_core) / Rs
        hx = self._h(x, p)
        return 2 * self._rescale_density(Rs, r_core)* rho0 * Rs ** 3 * hx

    def cBurkGamma(self, R, Rs, rho0, r_core, ax_x, ax_y):
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

        p = float(r_core) * Rs ** -1

        gx = self.G(x, p)
        Fx = self.F(x, p)

        #a = rho0*Rs*(gx * x ** -4 + 4*np.pi * Rs ** 2 * Fx * x**-2)
        #print(a)
        a = rho0*Rs*(gx/x**2 - Fx)  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x

        return a * (ax_y ** 2 - ax_x ** 2) / R ** 2, -a * 2 * (ax_x * ax_y) / R ** 2

    def mass_2d(self,R,Rs,rho0,r_core):

        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """

        x = R / Rs
        p = r_core / Rs
        gx = self._g(x,p)
        m_2d = 4 * rho0 * Rs * R ** 2 * gx / x ** 2 * np.pi
        return m_2d * self._rescale_density(Rs, r_core)

    def _h(self, X, p):

        """
        a horrible expression for the integral to compute potential

        :param x: R/Rs
        :param p: t/Rs
        :type x: float >0
        """
        pass

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

import matplotlib.pyplot as plt
from scipy.integrate import quad
from lenstronomy.LensModel.Profiles.nfw import NFW
if False:

    b = coreBurkert()
    n = NFW()
    Rs = 10
    r_core = 4
    x = np.linspace(0.01*Rs,30*Rs,1000)

    kapb = b.density_2d(x,0,Rs,1,r_core)
    kapn = n.density_2d(x,0,Rs,1)
    plt.loglog(x / Rs, kapb, color='k')
    plt.loglog(x / Rs, kapn, color='r')
    plt.show()

if False:
    b = coreBurkert(Rmax_norm=10)
    n = NFW()
    Rs = 20
    r_core =7
    x = np.linspace(0.01 * Rs, 100 * Rs, 1000)

    dx, _ = b.derivatives(x, 0, Rs, 1, r_core)
    dxn, _ = n.derivatives(x, 0, Rs, 1)
    plt.loglog(x / Rs, dx, color='k')
    plt.loglog(x / Rs, dxn, color='r')
    plt.show()

if False:
    b = coreBurkert()
    n = NFW()
    Rs = 10
    r_core = 5
    print(b.mass_3d(5*Rs, Rs, 1, 0.4*Rs))
    print(n.mass_3d(5*Rs, Rs, 1))

if False:
    b = coreBurkert(Rmax_norm=10)
    n = NFW()
    Rs = 20
    r_core = 3
    x = np.linspace(0.01 * Rs, 100 * Rs, 1000)

    xx, yy, xy = b.hessian(x, 0, Rs, 1, r_core)
    xxn, yyn, xyn = n.hessian(x, 0, Rs, 1)

    plt.loglog(x / Rs, yy, color='k')
    #plt.loglog(x / Rs, b.density_2d(x, 0 , Rs, 1, r_core), color='r')
    plt.loglog(x / Rs, yyn, color='r')
    plt.show()
