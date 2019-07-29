from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.constants as const


class Background(object):
    """
    class to compute cosmological distances
    """
    def __init__(self, cosmo=None):
        """

        :param cosmo: instance of astropy.cosmology
        :return: Background class with instance of astropy.cosmology
        """
        from astropy.cosmology import default_cosmology

        if cosmo is None:
            cosmo = default_cosmology.get()
        self.cosmo = cosmo

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        :param z: redshift
        :return: scale factor
        """
        return 1./(1+z)

    def D_xy(self, z_observer, z_source):
        """

        :param z_observer: observer
        :param z_source: source
        :return: angular diamter distance in units of Mpc
        """
        D_xy = self.cosmo.angular_diameter_distance_z1z2(z_observer, z_source)
        return D_xy.value

    def D_dt(self, z_lens, z_source):
        """
        time-delay distance

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :return: time-delay distance in units of Mpc
        """
        return self.D_xy(0, z_lens) * self.D_xy(0, z_source) / self.D_xy(z_lens, z_source) * (1 + z_lens)

    def T_xy(self, z_observer, z_source):
        """

        :param z_observer: observer
        :param z_source: source
        :return: transverse comoving distance in units of Mpc
        """
        D_xy = self.D_xy(z_observer, z_source)
        T_xy = D_xy * (1 + z_source)
        return T_xy

    @property
    def rho_crit(self):
        """
        critical density
        :return: value in M_sol/Mpc^3
        """
        h = self.cosmo.H(0).value / 100.
        return 3 * h ** 2 / (8 * np.pi * const.G) * 10 ** 10 * const.Mpc / const.M_sun


