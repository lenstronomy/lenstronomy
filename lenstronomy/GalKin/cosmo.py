import lenstronomy.Util.constants as const
import numpy as np


class Cosmo(object):
    """
    cosmological quantities
    """
    def __init__(self, kwargs_cosmo):
        self.D_d = float(kwargs_cosmo['D_d'])  # angular diamter distance from observer to deflector in physical Mpc
        self.D_s = float(kwargs_cosmo['D_s'])  # angular diamter distance from observer to source in physical Mpc
        self.D_ds = float(kwargs_cosmo['D_ds'])  # angular diamter distance from deflector to source in physical Mpc

    def arcsec2phys_lens(self, theta):
        """
        converts are seconds to physical units on the deflector
        :param theta:
        :return:
        """
        return theta * const.arcsec * self.D_d

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        """
        const_SI = const.c**2 / (4*np.pi * const.G)  #c^2/(4*pi*G) in units of [kg/m]
        conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
        pre_const = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
        Epsilon_Crit = self.D_s/(self.D_d*self.D_ds) * pre_const #[M_sun/Mpc^2]
        return Epsilon_Crit