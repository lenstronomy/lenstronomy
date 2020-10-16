__author__ = 'sibirrer'

# this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
# the potential therefore is its integral

import numpy as np
from lenstronomy.Util import  constants as const
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ['NFWVirTrunc']


class NFWVirTrunc(LensProfileBase):
    """
    this class contains functions concerning the NFW profile that is sharply truncated at the virial radius
    https://arxiv.org/pdf/astro-ph/0304034.pdf

    relation are: R_200 = c * Rs
    """
    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy cosmology instance
        """

        if cosmo is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        super(NFWVirTrunc, self).__init__()

    def kappa(self, theta, logM, c):
        """
        projected surface brightness

        :param theta: radial angle from the center of the profile
        :param logM: log_10 halo mass in physical units of M_sun
        :param c: concentration of the halo; r_vir = c * r_s
        :return: convergence at theta
        """
        M = 10. ** logM
        theta_vir = self._lens_cosmo.nfw_M_theta_vir(M)
        #r_vir = theta_vir * self._lens_cosmo.D_d * const.arcsec  # physical Mpc
        #print(r_vir, 'r_vir')
        x = c * theta / theta_vir
        f = self._f(c)
        return M / self._lens_cosmo.sigma_crit_angle * c ** 2 * f / (2 * np.pi * theta_vir ** 2) * self._G(x, c)

    def _G(self, x, c):
        """
        # G(x) https://arxiv.org/pdf/astro-ph/0209167.pdf equation 27

        :param x: scaled projected radius with c * theta / theta_vir
        :param c: oncentration of the halo; r_vir = c * r_s
        :return: G(x)
        """
        s = 0.000001
        if isinstance(x, int) or isinstance(x, float):
            if x < 1:
                x = max(s, x)
                a = - np.sqrt(c**2 - x**2) / (1 - x**2) / (1 + c) + 1 / (1 - x**2)**(3./2) * np.arccosh((x**2 + c) / (x * (1 + c)))
            elif x == 1:
                a = np.sqrt(c**2 - 1) / (3 * (1 + c)) * (1 + 1 / (c + 1.))
            elif x <= c:  # X > 1:
                a = - np.sqrt(c ** 2 - x ** 2) / (1 - x ** 2) / (1 + c) - 1 / (x ** 2 - 1) ** (3. / 2) * np.arccos(
                    (x ** 2 + c) / (x * (1 + c)))
            else:
                a = 0

        else:
            a = np.zeros_like(x)
            x[x <= s] = s
            x_ = x[x < 1]
            a[x < 1] = - np.sqrt(c**2 - x_**2) / ((1 - x_**2) * (1 + c)) + 1 / (1 - x_**2)**(3./2) * np.arccosh((x_**2 + c) / (x_ * (1 + c)))
            a[x == 1] = np.sqrt(c**2 - 1) / (3 * (1 + c)) * (1 + 1 / (c + 1.))
            x_ = x[(x > 1) & (x <= c)]
            a[(x > 1) & (x <= c)] = - np.sqrt(c ** 2 - x_ ** 2) / (1 - x_ ** 2) / (1 + c) - 1 / (x_ ** 2 - 1) ** (3. / 2) * np.arccos(
                    (x_ ** 2 + c) / (x_ * (1 + c)))
            #a[x > c] = 0
        return a

    def _f(self, c):
        """

        :param c: concentration
        :return: dimensionless normalization of Halo mass
        """
        return 1. / (np.log(1 + c) - c / (1 + c))

# https://arxiv.org/pdf/astro-ph/0304034.pdf equation 17 for shear

