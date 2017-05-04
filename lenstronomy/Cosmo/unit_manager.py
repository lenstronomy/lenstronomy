__author__ = 'sibirrer'

#this file contains a class to convert units

import numpy as np
import lenstronomy.Cosmo.constants as const
from lenstronomy.Cosmo.cosmo_properties import CosmoProp


class UnitManager(object):
    """
    class to manage the unit conversions
    """
    def __init__(self, z_lens, z_source):
        """
        :param CosmoProp: instance of class CosmoProp
        :param StrongLensSystem: instance of a StrongLensSystem class
        """
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmoProp = CosmoProp(z_lens=self.z_lens, z_source=self.z_source)
        """
        test test
        """

    def _linear_convert(self, input, converting_const, center=None):
        """
        general function for linear conversion (translation and stretching) for arbitrary dimensions of input
        test
        """
        if not center is None:
            input = input - center #translation
        return input*converting_const

    def phys2arcsec_lens(self, phys):
        # convert comoving Mpc/h into arc seconds
        return phys/self.cosmoProp.dist_OL/const.arcsec

    def arcsec2phys_lens(self, arcsec):
        return arcsec*const.arcsec*self.cosmoProp.dist_OL

    def arcsec2phys_source(self, arcsec):
        return arcsec * const.arcsec * self.cosmoProp.dist_OS

    def kappa2proj_mass(self, kappa):
        # convert convergence to projected mass M_sun/Mpc^2
        return kappa * self.cosmoProp.epsilon_crit

    def mass_in_phi_E(self, theta_E):
        # mass within Einstein radius (area * epsilon crit) [M_sun]
        mass = self.arcsec2phys_lens(theta_E) ** 2 * np.pi * self.cosmoProp.epsilon_crit
        return mass

    def mass_in_phi_E_2(self, theta_E):
        # mass within Einstein radius (area * epsilon crit) [M_sun]
        mass = (theta_E * const.arcsec) ** 2 * const.c ** 2 / (4 * const.G) / self.cosmoProp.dist_LS * (self.cosmoProp.dist_OL * self.cosmoProp.dist_OS) / const.M_sun * const.Mpc
        return mass

    def mass_in_coin(self, theta_E):
        """

        :param theta_E: Einstein radius
        :return: mass in coin calculated in mean density of the universe
        """
        chi_L = self.cosmoProp.trans_dist_L
        chi_S = self.cosmoProp.trans_dist_L
        return 1./3 * np.pi * (chi_L * theta_E * const.arcsec) ** 2 * chi_S * self.cosmoProp.rho_crit * self.cosmoProp.cosmo.background._omega_m_a(a=1.0)[0] #[M_sun/Mpc**3]

    def time_delay_units(self, delay_arcsec, kappa_ext=0):
        """

        :param delay_unitless: in units of arcsec^2 (Fermat potential)
        :param kappa_ext: unit less
        :return: time delay in days
        """
        D_dt_model = self.cosmoProp.D_dt_model
        D_dt = D_dt_model/(1. - kappa_ext) * const.Mpc  # eqn 7 in Suyu et al.
        return D_dt / const.c * delay_arcsec / const.day_s * const.arcsec**2  # * self.arcsec2phys_lens(1.)**2

