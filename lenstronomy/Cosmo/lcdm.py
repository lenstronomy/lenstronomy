__author__ = 'sibirrer'

from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ['LCDM']


class LCDM(object):
    """
    Flat LCDM cosmology background with free Hubble parameter and Omega_m at fixed lens redshift configuration
    """

    def __init__(self, z_lens, z_source, flat=True):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param flat: bool, if True, flat universe is assumed
        """
        self.z_lens = z_lens
        self.z_source = z_source
        self._flat = flat

    def _get_cosom(self, H_0, Om0, Ode0=None):
        """

        :param H_0:
        :param Om0:
        :param Ode0:
        :return:
        """
        if self._flat is True:
            cosmo = FlatLambdaCDM(H0=H_0, Om0=Om0)
        else:
            cosmo = LambdaCDM(H0=H_0, Om0=Om0, Ode0=Ode0)
        lensCosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=cosmo)
        return lensCosmo

    def D_d(self, H_0, Om0, Ode0=None):
        """
        angular diameter to deflector
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.dd

    def D_s(self, H_0, Om0, Ode0=None):
        """
        angular diameter to source
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.ds

    def D_ds(self, H_0, Om0, Ode0=None):
        """
        angular diameter from deflector to source
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.dds

    def D_dt(self, H_0, Om0, Ode0=None):
        """
        time delay distance
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.ddt
