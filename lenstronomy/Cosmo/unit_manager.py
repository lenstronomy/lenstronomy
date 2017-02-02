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

    def mass_in_phi_E(self, phi_E):
        # mass within Einstein radius (area * epsilon crit) [M_sun]
        mass = self.arcsec2phys_lens(phi_E)**2 * np.pi * self.cosmoProp.epsilon_crit
        return mass

    def mass_in_phi_E_2(self, phi_E):
        # mass within Einstein radius (area * epsilon crit) [M_sun]
        mass = (phi_E*const.arcsec)**2 * const.c**2/(4*const.G)/self.cosmoProp.dist_LS*(self.cosmoProp.dist_OL * self.cosmoProp.dist_OS)/const.M_sun*const.Mpc
        return mass

    def mass_in_coin(self, phi_E):
        """

        :param phi_E: Einstein radius
        :return: mass in coin calculated in mean density of the universe
        """
        chi_L = self.cosmoProp.trans_dist_L
        chi_S = self.cosmoProp.trans_dist_L
        return 1./3 * np.pi * (chi_L * phi_E * const.arcsec)**2 * chi_S * self.cosmoProp.rho_crit * self.cosmoProp.cosmo.background._omega_m_a(a=1.0)[0] #[M_sun/Mpc**3]

    def force_between_clumps(self, phi_E_1, phi_E_2, r_angle):
        """

        :param phi_E_1: Einstein radius of first clump [arcsec]
        :param phi_E_2: Einstein radius of second clump [arcsec]
        :param r_angle: projected distance between clumps [arcsec]
        :return: acceleration [m/s^2]
        """
        mass_1_M_sun = self.mass_in_phi_E(phi_E_1) # mass of first clump in M_sun
        mass_2_M_sun = self.mass_in_phi_E(phi_E_2) # mass of second clump in M_sun
        r_mpc = self.arcsec2phys_lens(r_angle) # distance between clumps in Mpc
        F = const.G * mass_1_M_sun * mass_2_M_sun / r_mpc**2 * const.M_sun**2 / const.Mpc**2
        return F

    def estimated_dipole(self, phi_E_1, phi_E_2, r_angle):
        """
        estimate deflection angle from dipole
        :param phi_E_1:
        :param phi_E_2:
        :param r_angle:
        :return:
        """
        F = self.force_between_clumps(phi_E_1, phi_E_2, r_angle)
        M1 = self.mass_in_phi_E(phi_E_1) * const.M_sun # mass of first clump in kg
        M2 = self.mass_in_phi_E(phi_E_2) * const.M_sun # mass of first clump in kg
        r_m = self.arcsec2phys_lens(r_angle) * const.Mpc
        deflection_eff = 4 * F * r_m/(M1 + M2)/const.c**2
        alpha_eff = deflection_eff * self.cosmoProp.dist_LS/self.cosmoProp.dist_OS
        return alpha_eff/const.arcsec

    def mass_in_dipole(self, coupling, phi_E, numPix=100):
        """
        estimates effective mass in dipole within the Einstein radius of the lens
        :param coupling:
        :param phi_E:
        :return:
        """
        return coupling/2.*np.pi*phi_E * self.arcsec2phys_lens(1)**2 * self.cosmoProp.epsilon_crit

