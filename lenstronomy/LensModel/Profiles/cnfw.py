__author__ = 'dgilman'

import numpy as np
import warnings


class CNFW(object):
    """
    this class contains functions concerning the truncated NFW profile with a truncation function (r_core^2)*(r^2+r_core^2)

    relation are: R_200 = c * Rs

    """
    param_names = ['Rs', 'theta_Rs', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'theta_Rs': 0, 'r_core': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'theta_Rs': 10, 'r_core': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpol=True, num_interp_X=1000, max_interp_X=10):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        """
        self._interpol = interpol
        self._max_interp_X = max_interp_X
        self._num_interp_X = num_interp_X

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
        #f_ = self.nfwPot(R, Rs, rho0_input, r_core)

        #return f_

    def _nfw_func_1(self, x):

        if isinstance(x, np.ndarray):
            func = np.ones_like(x) * (1. / 3)
            inds1 = np.where(x > 1)
            inds2 = np.where(x < 1)
            u = np.sqrt(x[inds1] ** 2 - 1)
            func[inds1] = u ** -1 * np.arctan(u)
            u = np.sqrt(1 - x[inds2] ** 2)
            func[inds2] = u ** -1 * np.arctanh(u)
            return func
        else:
            if x == 1:
                return 1./3
            elif x > 1:
                u = np.sqrt(x ** 2 - 1)
                return u ** -1 * np.arctan(u)
            else:
                u = np.sqrt(1 - x**2)
                return u**-1 * np.arctanh(u)

    def _nfw_func_2(self, x, q):

        if isinstance(x, np.ndarray):
            func = np.ones_like(x)
            inds1 = np.where(x > q)
            inds2 = np.where(x < q)

            u = np.sqrt(x[inds1] ** 2 - q**2)
            func[inds1] = u ** -1 * np.arctan(q * u**-1)
            u = np.sqrt(q ** 2 - x[inds2] ** 2)
            func[inds2] = u ** -1 * np.arctanh(u * q**-1)

            return func

        else:
            if x > q:
                u = np.sqrt(x ** 2 - 1)
                return u ** -1 * np.arctan(q*u**-1)
            else:
                u = np.sqrt(1 - x**2)
                return u**-1 * np.arctanh(u*q**-1)

    def _nfw_func_3(self, x, q):

        if isinstance(x, np.ndarray):

            func = np.ones_like(x)
            inds1 = np.where(x > q)
            func[inds1] = np.sqrt(x[inds1]**2 - q**2) ** -1
            inds2 = np.where(x <= q)
            func[inds2] = q**2 - x[inds2]**2

            return func * (-np.pi * q * (x**2 -1))

        else:

            if x > q:
                return (x **2 - q ** 2) ** -0.5
            else:
                return q**2 - x**2

    def _F(self, x, q):

        prefactor_inv = (q-1) ** 2 * (x**2 - 1)

        func = (self._nfw_func_3(x, q) - 2*(q - 1) + 2*(-1 * q*x**2) * \
               self._nfw_func_1(x) + 2*q*(x**2-1)*self._nfw_func_2(x,q)) * prefactor_inv**-1

        return func

    def derivatives(self, x, y, Rs=None, theta_Rs=None, r_core=None, center_x=0, center_y=0):

        rho0_input = self._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, r_core, x_, y_)
        #return f_x, f_y

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
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, r_core, x_, y_)
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
        q = r_core * Rs ** -1
        X = R * Rs **-1
        return rho0*((1+q*X)*(1+X)**2)**-1

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
        q = r_core * Rs ** -1
        x = R * Rs ** -1
        Fx = self._F(x, q)
        return 2 * rho0 * Rs * Fx

    def mass_3d_infinity(self, R, Rs, rho0, r_core):
        """
        mass enclosed a 3d sphere or radius r

        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        q = r_core * Rs ** -1
        x = R * Rs ** -1

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
        tau = float(r_core) / Rs
        hx = self._h(x, tau)
        return 2 * rho0 * Rs ** 3 * hx

    def nfwAlpha(self, R, Rs, rho0, r_core, ax_x, ax_y):
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
        tau = float(r_core) / Rs
        gx = self._G(x, tau)
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * ax_x, a * ax_y

    def nfwGamma(self, R, Rs, rho0, r_core, ax_x, ax_y):
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

        tau = float(r_core) * Rs ** -1

        gx = self._G(x, tau)
        Fx = self._F(x, tau)

    def mass_2d(self,R,Rs,rho0,r_core):

        """
        analytic solution of the projection integral
        (convergence)

        :param x: R/Rs
        :type x: float >0
        """

        x = R / Rs
        tau = r_core / Rs
        gx = self._G(x,tau)
        m_2d = 4 * rho0 * Rs * R ** 2 * gx / x ** 2 * np.pi
        return m_2d


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
n= CNFW()
x,q = np.linspace(0.01,0.8,1000), 0.4
f = n._F(x,q)
plt.loglog(x, f); plt.show()
