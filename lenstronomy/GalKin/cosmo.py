import lenstronomy.Util.constants as const
import numpy as np

__all__ = ['Cosmo']


class Cosmo(object):
    """
    cosmological quantities
    """
    def __init__(self, d_d, d_s, d_ds):
        """

        :param d_d: angular diameter distance to the deflector
        :param d_s: angular diameter distance to the source
        :param d_ds: angular diameter distance between deflector and source
        """
        if d_ds <= 0 or d_s <= 0 or d_d <=0:
            raise ValueError('input angular diameter distances Dd: %s, Ds: %s, Dds: %s are not suppored for a lens'
                             ' model!' % (d_d, d_s, d_ds))
        self.dd = float(d_d)  # angular diameter distance from observer to deflector in physical Mpc
        self.ds = float(d_s)  # angular diameter distance from observer to source in physical Mpc
        self.dds = float(d_ds)  # angular diameter distance from deflector to source in physical Mpc

    def arcsec2phys_lens(self, theta):
        """
        converts are seconds to physical units on the deflector
        :param theta: angle observed on the sky in units of arc seconds
        :return: pyhsical distance of the angle in units of Mpc
        """
        return theta * const.arcsec * self.dd

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        """
        const_SI = const.c**2 / (4*np.pi * const.G)  # c^2/(4*pi*G) in units of [kg/m]
        conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
        pre_const = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
        Epsilon_Crit = self.ds / (self.dd * self.dds) * pre_const  # [M_sun/Mpc^2]
        return Epsilon_Crit
