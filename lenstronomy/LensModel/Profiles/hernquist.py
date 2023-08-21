import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Hernquist']


class Hernquist(LensProfileBase):
    """Class to compute the Hernquist 1990 model, which is in 3d: rho(r) = rho0 / (r/Rs
    * (1 + (r/Rs))**3)

    in lensing terms, the normalization parameter 'sigma0' is defined such that the deflection at projected RS leads to
    alpha = 2./3 * Rs * sigma0

    Examples for converting angular to physical mass units
    ------------------------------------------------------

    >>> from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    >>> lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)

    Here we compute the angular scale of Rs on the sky (in arc seconds) and the deflection the normalization sigma0 from
    the total stellar mass in M_sol and Rs in [Mpc]:

    >>> sigma0, rs_angle = lens_cosmo.hernquist_phys2angular(mass=10**11, rs=0.02)

    And here we perform the inverse calculation given Rs_angle and alpha_Rs to return the physical halo properties.

    >>> m_tot, rs = lens_cosmo.hernquist_angular2phys(sigma0=sigma0 rs_angle=rs_angle)

    The lens model calculation uses angular units as arguments! So to execute a deflection angle calculation one uses

    >>> from lenstronomy.LensModel.Profiles.hernquist import Hernquist
    >>> hernquist = Hernquist()
    >>> alpha_x, alpha_y = hernquist.derivatives(x=1, y=1, Rs=rs_angle, sigma0=sigma0, center_x=0, center_y=0)
    """
    _diff = 0.00001
    _s = 0.00001
    param_names = ['sigma0', 'Rs', 'center_x', 'center_y']
    lower_limit_default = {'sigma0': 0, 'Rs': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'sigma0': 100, 'Rs': 100, 'center_x': 100, 'center_y': 100}

    @staticmethod
    def density(r, rho0, Rs):
        """Computes the 3-d density.

        :param r: 3-d radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: density at radius r
        """
        rho = rho0 / (r/Rs * (1 + (r/Rs))**3)
        return rho

    def density_lens(self, r, sigma0, Rs):
        """Density as a function of 3d radius in lensing parameters This function
        converts the lensing definition sigma0 into the 3d density.

        :param r: 3d radius
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: enclosed mass in 3d
        """
        rho0 = self.sigma2rho(sigma0, Rs)
        return self.density(r, rho0, Rs)

    def density_2d(self, x, y, rho0, Rs, center_x=0, center_y=0):
        """Projected density along the line of sight at coordinate (x, y)

        :param x: x-coordinate
        :param y: y-coordinate
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: projected density
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        X = r/Rs
        sigma0 = self.rho2sigma(rho0, Rs)
        if isinstance(X, int) or isinstance(X, float):
            if X == 1:
                X = 1.000001
        else:
            X[X == 1] = 1.000001
        sigma = sigma0 / (X**2-1)**2 * (-3 + (2+X**2)*self._F(X))
        return sigma

    @staticmethod
    def mass_3d(r, rho0, Rs):
        """Mass enclosed a 3d sphere or radius r.

        :param r: 3-d radius within the mass is integrated (same distance units as
            density definition)
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: enclosed mass
        """
        mass_3d = 2*np.pi*Rs**3*rho0 * r**2/(r + Rs)**2
        return mass_3d

    def mass_3d_lens(self, r, sigma0, Rs):
        """Mass enclosed a 3d sphere or radius r for lens parameterisation This function
        converts the lensing definition sigma0 into the 3d density.

        :param r: radius
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: enclosed mass in 3d
        """
        rho0 = self.sigma2rho(sigma0, Rs)
        return self.mass_3d(r, rho0, Rs)

    def mass_2d(self, r, rho0, Rs):
        """Mass enclosed projected 2d sphere of radius r.

        :param r: projected radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: mass enclosed 2d projected radius
        """

        sigma0 = self.rho2sigma(rho0, Rs)
        return self.mass_2d_lens(r, sigma0, Rs)

    def mass_2d_lens(self, r, sigma0, Rs):
        """Mass enclosed projected 2d sphere of radius r Same as mass_2d but with input
        normalization in units of projected density.

        :param r: projected radius
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: mass enclosed 2d projected radius
        """
        X = r/Rs
        alpha_r = 2*sigma0 * Rs * X * (1-self._F(X)) / (X**2-1)
        mass_2d = alpha_r * r * np.pi
        return mass_2d

    @staticmethod
    def mass_tot(rho0, Rs):
        """Total mass within the profile :param rho0: density normalization :param Rs:
        Hernquist radius :return: total mass within profile."""
        m_tot = 2*np.pi*rho0*Rs**3
        return m_tot

    def function(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """Lensing potential.

        :param x: x-coordinate position (units of angle)
        :param y: y-coordinate position (units of angle)
        :param sigma0: normalization parameter defined such that the deflection at
            projected RS leads to alpha = 2./3 * Rs * sigma0
        :param Rs: Hernquist radius in units of angle
        :param center_x: x-center of the profile (units of angle)
        :param center_y: y-center of the profile (units of angle)
        :return: lensing potential at (x,y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r = np.maximum(r, self._s)
        X = r / Rs
        f_ = sigma0 * Rs ** 2 * (np.log(X ** 2 / 4.) + 2 * self._F(X))
        return f_

    def derivatives(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """

        :param x: x-coordinate position (units of angle)
        :param y: y-coordinate position (units of angle)
        :param sigma0: normalization parameter defined such that the deflection at projected RS leads to
         alpha = 2./3 * Rs * sigma0
        :param Rs: Hernquist radius in units of angle
        :param center_x: x-center of the profile (units of angle)
        :param center_y: y-center of the profile (units of angle)
        :return: derivative of function (deflection angles in x- and y-direction)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r = np.maximum(r, self._s)
        X = r/Rs
        if isinstance(r, int) or isinstance(r, float):
            # f = (1 - self._F(X)) / (X ** 2 - 1)  # this expression is 1/3 for X=1
            if X == 1:
                f = 1./3
            else:
                f = (1 - self._F(X)) / (X ** 2 - 1)
        else:
            f = np.empty_like(X)
            f[X == 1] = 1./3
            X_ = X[X != 1]
            f[X != 1] = (1 - self._F(X_)) / (X_ ** 2 - 1)
        alpha_r = 2 * sigma0 * Rs * f * X
        f_x = alpha_r * x_/r
        f_y = alpha_r * y_/r
        return f_x, f_y

    def hessian(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """Hessian terms of the function.

        :param x: x-coordinate position (units of angle)
        :param y: y-coordinate position (units of angle)
        :param sigma0: normalization parameter defined such that the deflection at
            projected RS leads to alpha = 2./3 * Rs * sigma0
        :param Rs: Hernquist radius in units of angle
        :param center_x: x-center of the profile (units of angle)
        :param center_y: y-center of the profile (units of angle)
        :return: df/dxdx, df/dxdy, df/dydx, df/dydy
        """
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff / 2, y, sigma0, Rs, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff / 2, sigma0, Rs, center_x, center_y)

        alpha_ra_dx_, alpha_dec_dx_ = self.derivatives(x - diff / 2, y, sigma0, Rs, center_x, center_y)
        alpha_ra_dy_, alpha_dec_dy_ = self.derivatives(x, y - diff / 2, sigma0, Rs, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra_dx_) / diff
        f_xy = (alpha_ra_dy - alpha_ra_dy_) / diff
        f_yx = (alpha_dec_dx - alpha_dec_dx_) / diff
        f_yy = (alpha_dec_dy - alpha_dec_dy_) / diff
        return f_xx, f_xy, f_yx, f_yy

    @staticmethod
    def rho2sigma(rho0, Rs):
        """Converts 3d density into 2d projected density parameter :param rho0: 3d
        density normalization of Hernquist model :param Rs: Hernquist radius :return:
        sigma0 defined quantity in projected units."""
        return rho0 * Rs

    @staticmethod
    def sigma2rho(sigma0, Rs):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter :param sigma0: density defined quantity in projected units :param Rs:
        Hernquist radius :return: rho0 the 3d density normalization of Hernquist
        model."""
        return sigma0 / Rs

    def _F(self, X):
        """
        function 48 in https://arxiv.org/pdf/astro-ph/0102341.pdf
        :param X: r/rs
        :return: F(X)
        """
        c = self._s
        if isinstance(X, int) or isinstance(X, float):
            X = max(X, c)
            if 0 < X < 1:
                a = 1. / np.sqrt(1 - X ** 2) * np.arctanh(np.sqrt(1 - X**2))
            elif X == 1:
                a = 1.
            elif X > 1:
                a = 1. / np.sqrt(X ** 2 - 1) * np.arctan(np.sqrt(X**2 - 1))
            else:  # X == 0:
                a = 1. / np.sqrt(1 - c ** 2) * np.arctanh(np.sqrt((1 - c ** 2)))

        else:
            a = np.empty_like(X)
            X[X < c] = c
            x = X[X < 1]
            a[X < 1] = 1 / np.sqrt(1 - x ** 2) * np.arctanh(np.sqrt((1 - x**2)))

            # x = X[X == 1]
            a[X == 1] = 1.

            x = X[X > 1]
            a[X > 1] = 1 / np.sqrt(x ** 2 - 1) * np.arctan(np.sqrt(x**2 - 1))
            # a[X>y] = 0
        return a

    def grav_pot(self, x, y, rho0, Rs, center_x=0, center_y=0):
        """#TODO decide whether these functions are needed or not

        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x: x-coordinate position (units of angle)
        :param y: y-coordinate position (units of angle)
        :param rho0: density normalization parameter of Hernquist profile
        :param Rs: Hernquist radius in units of angle
        :param center_x: x-center of the profile (units of angle)
        :param center_y: y-center of the profile (units of angle)
        :return: gravitational potential at projected radius
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        M = self.mass_tot(rho0, Rs)
        pot = M / (r + Rs)
        return pot
