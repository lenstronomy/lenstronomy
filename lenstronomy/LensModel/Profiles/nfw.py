__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frenk-White profile
import numpy as np
import scipy.interpolate as interp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['NFW']


class NFW(LensProfileBase):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    The definition of 'Rs' is in angular (arc second) units and the normalization is put in in regards to a deflection
    angle at 'Rs' - 'alpha_Rs'. To convert a physical mass and concentration definition into those lensing quantities
    for a specific redshift configuration and cosmological model, you can find routines in lenstronomy.Cosmo.lens_cosmo.py

    Examples for converting angular to physical mass units
    ------------------------------------------------------
    >>> from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    >>> lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)
    >>> Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=10**13, c=6)
    >>> rho0, Rs, c, r200, M200 = lens_cosmo.nfw_angle2physical(Rs_angle=Rs_angle, alpha_Rs=alpha_Rs)

    """
    param_names = ['Rs', 'alpha_Rs', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpol=False, num_interp_X=1000, max_interp_X=10):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated (returning zeros outside)
        """
        self._interpol = interpol
        self._max_interp_X = max_interp_X
        self._num_interp_X = num_interp_X
        super(NFW, self).__init__()

    def function(self, x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """
        
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential
        """
        rho0_input = self._alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        f_ = self.nfwPot(R, Rs, rho0_input)
        return f_

    def derivatives(self, x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW), which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_input = self._alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        f_x, f_y = self.nfwAlpha(R, Rs, rho0_input, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, center_x=0, center_y=0):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        rho0_input = self._alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        kappa = self.density_2d(R, 0, Rs, rho0_input)
        gamma1, gamma2 = self.nfwGamma(R, Rs, rho0_input, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def density(self, R, Rs, rho0):
        """
        three dimensional NFW profile

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :return: rho(R) density
        """
        return rho0/(R/Rs*(1+R/Rs)**2)

    def density_lens(self, r, Rs, alpha_Rs):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: 3d radios
        :param Rs: turn-over radius of NFW profile
        :param alpha_Rs: deflection at Rs
        :return: density rho(r)
        """
        rho0 = self._alpha2rho0(alpha_Rs, Rs)
        return self.density(r, Rs, rho0)

    def density_2d(self, x, y, Rs, rho0, center_x=0, center_y=0):
        """
        projected two dimensional NFW profile (kappa*Sigma_crit)

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
        R = np.sqrt(x_**2 + y_**2)
        x = R/Rs
        Fx = self.F_(x)
        return 2*rho0*Rs*Fx

    def mass_3d(self, R, Rs, rho0):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Rs:
        :param rho0:
        :return:
        """
        Rs = float(Rs)
        m_3d = 4. * np.pi * rho0 * Rs**3 * (np.log((Rs + R)/Rs) - R/(Rs + R))
        return m_3d

    def mass_3d_lens(self, r, Rs, alpha_Rs):
        """
        mass enclosed a 3d sphere or radius r
        :param R:
        :param Rs:
        :param alpha_Rs:
        :return:
        """
        rho0 = self._alpha2rho0(alpha_Rs, Rs)
        m_3d = self.mass_3d(r, Rs, rho0)
        return m_3d

    def mass_2d(self, R, Rs, rho0):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        x = R/Rs
        gx = self.g_(x)
        m_2d = 4*rho0*Rs*R**2*gx/x**2 * np.pi
        return m_2d

    def nfwPot(self, R, Rs, rho0):
        """

        lensing potential of NFW profile (Sigma_crit D_OL**2)

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
        x = R/Rs
        hx = self.h_(x)
        return 2*rho0*Rs**3*hx

    def nfwAlpha(self, R, Rs, rho0, ax_x, ax_y):
        """

        deflection angel of NFW profile (times Sigma_crit D_OL) along the projection to coordinate 'axis'

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
        R = np.maximum(R, 0.00000001)
        #if isinstance(R, int) or isinstance(R, float):
        #    R = max(R, 0.00000001)
        #else:
        #    R[R <= 0.00000001] = 0.00000001
        x = R/Rs
        gx = self.g_(x)
        a = 4*rho0*Rs*R*gx/x**2/R
        return a*ax_x, a*ax_y

    def nfwGamma(self, R, Rs, rho0, ax_x, ax_y):
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
        R = np.maximum(R, c)
        #if isinstance(R, int) or isinstance(R, float):
        #    R = max(R, c)
        #else:
        #    R[R <= c] = c
        x = R/Rs
        gx = self.g_(x)
        Fx = self.F_(x)
        a = 2*rho0*Rs*(2*gx/x**2 - Fx)#/x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a*(ax_y**2-ax_x**2)/R**2, -a*2*(ax_x*ax_y)/R**2

    def F_(self, X):
        """
        computes h()

        :param X:
        :return:
        """
        if self._interpol:
            if not hasattr(self, '_F_interp'):
                x = np.linspace(0, self._max_interp_X, self._num_interp_X)
                F_x = self._F(x)
                self._F_interp = interp.interp1d(x, F_x, kind='linear', axis=-1, copy=False, bounds_error=False,
                                                 fill_value=0, assume_sorted=True)
            return self._F_interp(X)
        else:
            return self._F(X)

    def _F(self, X):
        """
        analytic solution of the projection integral

        :param x: R/Rs
        :type x: float >0
        """
        if isinstance(X, int) or isinstance(X, float):
            if X < 1 and X > 0:
                a = 1/(X**2-1)*(1-2/np.sqrt(1-X**2)*np.arctanh(np.sqrt((1-X)/(1+X))))
            elif X == 1:
                a = 1./3
            elif X > 1:
                a = 1/(X**2-1)*(1-2/np.sqrt(X**2-1)*np.arctan(np.sqrt((X-1)/(1+X))))
            else:  # X == 0:
                c = 0.0000001
                a = 1/(-1)*(1-2/np.sqrt(1)*np.arctanh(np.sqrt((1-c)/(1+c))))

        else:
            a = np.empty_like(X)
            x = X[(X < 1) & (X > 0)]
            a[(X < 1) & (X > 0)] = 1/(x**2-1)*(1-2/np.sqrt(1-x**2)*np.arctanh(np.sqrt((1-x)/(1+x))))

            a[X == 1] = 1./3.

            x = X[X > 1]
            a[X > 1] = 1/(x**2-1)*(1-2/np.sqrt(x**2-1)*np.arctan(np.sqrt((x-1)/(1+x))))
            # a[X>y] = 0

            c = 0.0000001
            a[X == 0] = 1/(-1)*(1-2/np.sqrt(1)*np.arctanh(np.sqrt((1-c)/(1+c))))
        return a

    def g_(self, X):
        """
        computes h()

        :param X:
        :return:
        """
        if self._interpol:
            if not hasattr(self, '_g_interp'):
                x = np.linspace(0, self._max_interp_X, self._num_interp_X)
                g_x = self._g(x)
                self._g_interp = interp.interp1d(x, g_x, kind='linear', axis=-1, copy=False, bounds_error=False,
                                                 fill_value=0, assume_sorted=True)
            return self._g_interp(X)
        else:
            return self._g(X)

    def _g(self, X):
        """

        analytic solution of integral for NFW profile to compute deflection angel and gamma

        :param x: R/Rs
        :type x: float >0
        """
        c = 0.000001
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(c, X)
                a = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            elif X == 1:
                a = 1 + np.log(1./2.)
            else:  # X > 1:
                a = np.log(X/2) + 1/np.sqrt(X**2-1)*np.arccos(1./X)

        else:
            a = np.empty_like(X)
            X[X <= c] = c
            x = X[X < 1]
            a[X < 1] = np.log(x/2.) + 1/np.sqrt(1-x**2)*np.arccosh(1./x)
            a[X == 1] = 1 + np.log(1./2.)
            x = X[X > 1]
            a[X > 1] = np.log(x/2) + 1/np.sqrt(x**2-1)*np.arccos(1./x)

        return a

    def h_(self, X):
        """
        computes h()

        :param X:
        :return:
        """
        if self._interpol:
            if not hasattr(self, '_h_interp'):
                x = np.linspace(0, self._max_interp_X, self._num_interp_X)
                h_x = self._h(x)
                self._h_interp = interp.interp1d(x, h_x, kind='linear', axis=-1, copy=False, bounds_error=False,
                                                 fill_value=0, assume_sorted=True)
            return self._h_interp(X)
        else:
            return self._h(X)

    def _h(self, X):
        """

        analytic solution of integral for NFW profile to compute the potential

        :param x: R/Rs
        :type x: float >0
        """
        c = 0.000001
        if isinstance(X, int) or isinstance(X, float):
            if X < 1:
                x = max(0.001, X)
                a = np.log(x/2.)**2 - np.arccosh(1./x)**2
            else:  # X >= 1:
                a = np.log(X/2.)**2 + np.arccos(1./X)**2
        else:
            a = np.empty_like(X)
            X[X <= c] = 0.000001
            x = X[X < 1]
            a[X < 1] = np.log(x/2.)**2 - np.arccosh(1./x)**2
            x = X[X >= 1]
            a[X >= 1] = np.log(x/2.)**2 + np.arccos(1./x)**2
        return a

    @staticmethod
    def _alpha2rho0(alpha_Rs, Rs):

        """
        convert angle at Rs into rho0
        """

        rho0 = alpha_Rs / (4. * Rs ** 2 * (1. + np.log(1. / 2.)))
        return rho0

    @staticmethod
    def _rho02alpha(rho0, Rs):

        """
        convert rho0 to angle at Rs

        :param rho0:
        :param Rs:
        :return:
        """

        alpha_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return alpha_Rs
