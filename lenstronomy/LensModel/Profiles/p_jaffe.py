import numpy as np


class PJaffe(object):
    """
    class to compute the DUAL PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION
    based on Eliasdottir (2013)
    """
    _s = 0.0001
    param_names = ['sigma0', 'Ra', 'Rs', 'center_x', 'center_y']
    lower_limit_default = {'sigma0': 0, 'Ra': 0, 'Rs': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'sigma0': 10, 'Ra': 100, 'Rs': 100, 'center_x': 100, 'center_y': 100}

    def density(self, r, rho0, Ra, Rs):
        """
        computes the density
        :param x:
        :param y:
        :param rho0:
        :param Ra:
        :param Rs:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        rho = rho0 / ((1 + (r / Ra) ** 2) * (1 + (r / Rs) ** 2))
        return rho

    def density_2d(self, x, y, rho0, Ra, Rs, center_x=0, center_y=0):
        """
        projected density
        :param x:
        :param y:
        :param rho0:
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma0 = self.rho2sigma(rho0, Ra, Rs)
        sigma = sigma0 * Ra * Rs / (Rs - Ra) * (1 / np.sqrt(Ra ** 2 + r ** 2) - 1 / np.sqrt(Rs ** 2 + r ** 2))
        return sigma

    def mass_3d(self, r, rho0, Ra, Rs):
        """
        mass enclosed a 3d sphere or radius r
        :param r:
        :param Ra:
        :param Rs:
        :return:
        """
        m_3d = 4 * np.pi * rho0 * Ra ** 2 * Rs ** 2 / (Rs ** 2 - Ra ** 2) * (Rs * np.arctan(r / Rs) - Ra * np.arctan(r / Ra))
        return m_3d

    def mass_3d_lens(self, r, sigma0, Ra, Rs):
        """

        :param r:
        :param sigma0:
        :param Ra:
        :param Rs:
        :return:
        """
        rho0 = self.sigma2rho(sigma0, Ra, Rs)
        return self.mass_3d(r, rho0, Ra, Rs)

    def mass_2d(self, r, rho0, Ra, Rs):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param Ra:
        :param Rs:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        sigma0 = self.rho2sigma(rho0, Ra, Rs)
        m_2d = 2 * np.pi * sigma0 * Ra * Rs / (Rs - Ra) * (np.sqrt(Ra ** 2 + r ** 2) - Ra - np.sqrt(Rs ** 2 + r ** 2) + Rs)
        return m_2d

    def mass_tot(self, rho0, Ra, Rs):
        """
        total mass within the profile
        :param rho0:
        :param Ra:
        :param Rs:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        sigma0 = self.rho2sigma(rho0, Ra, Rs)
        m_tot = 2 * np.pi * sigma0 * Ra * Rs
        return m_tot

    def grav_pot(self, x, y, rho0, Ra, Rs, center_x=0, center_y=0):
        """
        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x:
        :param y:
        :param rho0:
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        pot = 4 * np.pi * rho0 * Ra ** 2 * Rs ** 2 / (Rs ** 2 - Ra ** 2) * (Rs / r * np.arctan(r / Rs) - Ra / r * np.arctan(r / Ra)
                                                                            + 1. / 2 * np.log((Rs ** 2 + r ** 2) / (Ra ** 2 + r ** 2)))
        return pot

    def function(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        f_ = -2*sigma0 * Ra * Rs / (Rs - Ra) * (np.sqrt(Rs ** 2 + r ** 2) - np.sqrt(Ra ** 2 + r ** 2) + Ra * np.log(Ra + np.sqrt(Ra ** 2 + r ** 2)) - Rs * np.log(Rs + np.sqrt(Rs ** 2 + r ** 2)))
        return f_

    def derivatives(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        alpha_r = 2*sigma0 * Ra * Rs / (Rs - Ra) * self._f_A20(r / Ra, r / Rs)
        f_x = alpha_r * x_/r
        f_y = alpha_r * y_/r
        return f_x, f_y

    def hessian(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0: sigma0/sigma_crit
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        Ra, Rs = self._sort_ra_rs(Ra, Rs)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        gamma = sigma0 * Ra * Rs / (Rs - Ra) * (2 * (1. / (Ra + np.sqrt(Ra ** 2 + r ** 2)) - 1. / (Rs + np.sqrt(Rs ** 2 + r ** 2))) -
                                                (1 / np.sqrt(Ra ** 2 + r ** 2) - 1 / np.sqrt(Rs ** 2 + r ** 2)))
        kappa = sigma0 * Ra * Rs / (Rs - Ra) * (1 / np.sqrt(Ra ** 2 + r ** 2) - 1 / np.sqrt(Rs ** 2 + r ** 2))
        sin_2phi = -2*x_*y_/r**2
        cos_2phi = (y_**2 - x_**2)/r**2
        gamma1 = cos_2phi*gamma
        gamma2 = sin_2phi*gamma

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def _f_A20(self, r_a, r_s):
        """
        equation A20 in Eliasdottir (2013)
        :param r_a: r/Ra
        :param r_s: r/Rs
        :return:
        """
        return r_a/(1+np.sqrt(1 + r_a**2)) - r_s/(1+np.sqrt(1 + r_s**2))

    def rho2sigma(self, rho0, Ra, Rs):
        """
        converts 3d density into 2d projected density parameter
        :param rho0:
        :param Ra:
        :param Rs:
        :return:
        """
        return np.pi * rho0 * Ra * Rs / (Rs + Ra)

    def sigma2rho(self, sigma0, Ra, Rs):
        """

        :param sigma0:
        :param Ra:
        :param Rs:
        :return:
        """
        return (Rs + Ra) / Ra / Rs / np.pi * sigma0

    def _sort_ra_rs(self, Ra, Rs):
        if Ra >= Rs:
            Ra, Rs = Rs, Ra
        if Ra < 0.0001:
            Ra = 0.0001
        if Rs < Ra + 0.0001:
            Rs += 0.0001
        return Ra, Rs