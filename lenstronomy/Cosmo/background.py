from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp

__all__ = ['Background']


class Background(object):
    """
    class to compute cosmological distances
    """
    def __init__(self, cosmo=None, interp=False, **kwargs_interp):
        """

        :param cosmo: instance of astropy.cosmology
        :param interp: boolean, if True, uses interpolated cosmology to evaluate specific redshifts
        :param kwargs_interp: keyword arguments of CosmoInterp specifying the interpolation interval and maximum
        redshift
        :return: Background class with instance of astropy.cosmology
        """

        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        if interp:
            self.cosmo = CosmoInterp(cosmo, **kwargs_interp)
        else:
            self.cosmo = cosmo

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        :param z: redshift
        :return: scale factor
        """
        return 1./(1+z)

    def d_xy(self, z_observer, z_source):
        """

        :param z_observer: observer redshift
        :param z_source: source redshift
        :return: angular diameter distance in units of Mpc
        """
        D_xy = self.cosmo.angular_diameter_distance_z1z2(z_observer, z_source)
        return D_xy.value

    def ddt(self, z_lens, z_source):
        """
        time-delay distance

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :return: time-delay distance in units of proper Mpc
        """
        return self.d_xy(0, z_lens) * self.d_xy(0, z_source) / self.d_xy(z_lens, z_source) * (1 + z_lens)

    def T_xy(self, z_observer, z_source):
        """

        :param z_observer: observer
        :param z_source: source
        :return: transverse comoving distance in units of Mpc
        """
        D_xy = self.d_xy(z_observer, z_source)
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
