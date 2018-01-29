import numpy as np


class Hernquist(object):
    """
    class to compute the Hernquist 1990 model
    """
    _diff = 0.00001
    _s = 0.0001

    def density(self, r, rho0, Rs):
        """
        computes the density
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        rho = rho0 / (r/Rs * (1 + (r/Rs))**3)
        return rho

    def density_2d(self, x, y, rho0, Rs, center_x=0, center_y=0):
        """
        projected density
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
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

    def _F(self, X):
        """
        function 48 in https://arxiv.org/pdf/astro-ph/0102341.pdf
        :param X: r/rs
        :return:
        """
        c = 0.0000001
        if isinstance(X, int) or isinstance(X, float):
            X = max(X, c)
            if X < 1 and X > 0:
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

            x = X[X == 1]
            a[X == 1] = 1.

            x = X[X > 1]
            a[X > 1] = 1 / np.sqrt(x ** 2 - 1) * np.arctan(np.sqrt(x**2 - 1))
            # a[X>y] = 0
        return a

    def mass_3d(self, r, rho0, Rs):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param a:
        :param s:
        :return:
        """
        mass_3d = 2*np.pi*Rs**3*rho0 * r**2/(r + Rs)**2
        return mass_3d

    def mass_3d_lens(self, r, sigma0, Rs):
        """
        mass enclosed a 3d sphere or radius r for lens parameterisation
        :param sigma0:
        :param Rs:
        :return:
        """
        rho0 = self.sigma2rho(sigma0, Rs)
        return self.mass_3d(r, rho0, Rs)

    def mass_2d(self, r, rho0, Rs):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param a:
        :param s:
        :return:
        """

        sigma0 = self.rho2sigma(rho0, Rs)
        return self.mass_2d_lens(r, sigma0, Rs)

    def mass_2d_lens(self, r, sigma0, Rs):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        X = r/Rs
        alpha_r = 2*sigma0 * Rs * X * (1-self._F(X)) / (X**2-1)
        mass_2d = alpha_r * r * np.pi
        return mass_2d

    def mass_tot(self, rho0, Rs):
        """
        total mass within the profile
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        m_tot = 2*np.pi*rho0*Rs**3
        return m_tot

    def grav_pot(self, x, y, rho0, Rs, center_x=0, center_y=0):
        """
        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        M = self.mass_tot(rho0, Rs)
        pot = M / (r + Rs)
        return pot

    def function(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """
        lensing potential
        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        X = r / Rs
        f_ = sigma0 * Rs**2 * (np.log(X**2/4.) + 2*self._F(X))
        return f_

    def derivatives(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        X = r/Rs
        alpha_r = 2*sigma0 * Rs * X * (1-self._F(X)) / (X**2-1)
        f_x = alpha_r * x_/r
        f_y = alpha_r * y_/r
        return f_x, f_y

    def hessian(self, x, y, sigma0, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, sigma0, Rs,  center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, sigma0, Rs,  center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, sigma0, Rs,  center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        #f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff

        return f_xx, f_yy, f_xy

    def rho2sigma(self, rho0, Rs):
        """
        converts 3d density into 2d projected density parameter
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        return rho0 * Rs

    def sigma2rho(self, sigma0, Rs):
        """
        converts projected density parameter (in units of deflection) into 3d density parameter
        :param sigma0:
        :param Rs:
        :return:
        """
        return sigma0 / Rs