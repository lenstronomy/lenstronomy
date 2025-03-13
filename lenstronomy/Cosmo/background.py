__author__ = "sibirrer"

import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp

__all__ = ["Background"]


class Background(object):
    """Class to compute cosmological distances."""

    def __init__(self, cosmo=None, interp=False, **kwargs_interp):
        """

        :param cosmo: instance of astropy.cosmology
        :param interp: boolean, if True, uses interpolated cosmology to evaluate specific redshifts
        :param kwargs_interp: keyword arguments of CosmoInterp specifying the interpolation interval and maximum
         redshift
        :return: Background class with instance of astropy.cosmology
        """
        self.rhoc = 2.77536627e11  # critical density [h^2 M_sun Mpc^-3]
        if cosmo is None:
            from astropy.cosmology import default_cosmology

            cosmo = default_cosmology.get()
        if interp:
            self.cosmo = CosmoInterp(cosmo, **kwargs_interp)
        else:
            self.cosmo = cosmo

    @staticmethod
    def a_z(z):
        """Returns scale factor (a_0 = 1) for given redshift.

        :param z: redshift
        :return: scale factor
        """
        return 1.0 / (1 + z)

    def d_xy(self, z_observer, z_source):
        """

        :param z_observer: observer redshift
        :param z_source: source redshift
        :return: angular diameter distance in units of Mpc
        """
        D_xy = self.cosmo.angular_diameter_distance_z1z2(z_observer, z_source)
        return D_xy.value

    def ddt(self, z_lens, z_source):
        """Time-delay distance.

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :return: time-delay distance in units of proper Mpc
        """
        return (
            self.d_xy(0, z_lens)
            * self.d_xy(0, z_source)
            / self.d_xy(z_lens, z_source)
            * (1 + z_lens)
        )

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
        """Critical density.

        :return: value in M_sol/Mpc^3
        """
        h = self.cosmo.H(0).value / 100.0
        return 3 * h**2 / (8 * np.pi * const.G) * 10**10 * const.Mpc / const.M_sun

    def rho_crit_z(self, z):
        """Critical density of the universe at given redshift.

        :param z: redshift
        :return: critical densith in physical [M_sun/Mpc^3]
        """
        h = self.cosmo.H(0).value / 100.0
        return self.rhoc * (self.cosmo.efunc(z)) ** 2 * h**2

    def beta_double_source_plane(self, z_lens, z_source_1, z_source_2):
        """Model prediction of ratio of scaled deflection angles.

        :param z_lens: lens redshift
        :param z_source_1: source_1 redshift
        :param z_source_2: source_2 redshift
        :param cosmo: ~astropy.cosmology instance
        :return: beta
        """
        if z_source_1 == z_source_2:
            return 1
        ds1 = self.cosmo.angular_diameter_distance(z=z_source_1).value
        dds1 = self.cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_1).value
        ds2 = self.cosmo.angular_diameter_distance(z=z_source_2).value
        dds2 = self.cosmo.angular_diameter_distance_z1z2(z1=z_lens, z2=z_source_2).value
        beta = dds1 / ds1 * ds2 / dds2
        return beta

    def ddt_scaling(self, z_lens, z_source_1, z_source_2):
        """Scales the time-delay distance Ddt when given for one source redshift to a
        second source redshift.

        :param z_lens: deflector redshift
        :param z_source_1: source redshift of original Ddt
        :param z_source_2: new source redshift
        :return: Ddt to z_source_2
        """
        if z_source_1 == z_source_2:
            return 1
        return 1.0 / self.ddt(z_lens, z_source_1) * self.ddt(z_lens, z_source_2)
