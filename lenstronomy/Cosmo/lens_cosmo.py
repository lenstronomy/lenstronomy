__author__ = "sibirrer"

# this file contains a class to convert lensing and physical units

import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.background import Background
from lenstronomy.Cosmo.nfw_param import NFWParam

__all__ = ["LensCosmo"]


class LensCosmo(object):
    """Class to manage the physical units and distances present in a single plane lens
    with fixed input cosmology."""

    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy.cosmology instance
        """

        self.z_lens = z_lens
        self.z_source = z_source
        self.background = Background(cosmo=cosmo)
        self.nfw_param = NFWParam(cosmo=cosmo)

    @property
    def h(self):
        return self.background.cosmo.H(0).value / 100.0

    @property
    def dd(self):
        """

        :return: angular diameter distance to the deflector [Mpc]
        """
        return self.background.d_xy(0, self.z_lens)

    @property
    def ds(self):
        """

        :return: angular diameter distance to the source [Mpc]
        """
        return self.background.d_xy(0, self.z_source)

    @property
    def dds(self):
        """

        :return: angular diameter distance from deflector to source [Mpc]
        """
        return self.background.d_xy(self.z_lens, self.z_source)

    @property
    def ddt(self):
        """

        :return: time delay distance [Mpc]
        """
        return (1 + self.z_lens) * self.dd * self.ds / self.dds

    @property
    def sigma_crit(self):
        """Returns the critical projected lensing mass density in units of M_sun/Mpc^2.

        :return: critical projected lensing mass density
        """
        if not hasattr(self, "_sigma_crit_mpc"):
            const_SI = const.c**2 / (
                4 * np.pi * const.G
            )  # c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._sigma_crit_mpc = (
                self.ds / (self.dd * self.dds) * factor
            )  # [M_sun/Mpc^2]
        return self._sigma_crit_mpc

    @property
    def sigma_crit_angle(self):
        """Returns the critical surface density in units of M_sun/arcsec^2 (in physical
        solar mass units) when provided a physical mass per physical Mpc^2.

        :return: critical projected mass density
        """
        if not hasattr(self, "_sigma_crit_arcsec"):
            const_SI = const.c**2 / (
                4 * np.pi * const.G
            )  # c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._sigma_crit_arcsec = (
                self.ds / (self.dd * self.dds) * factor * (self.dd * const.arcsec) ** 2
            )  # [M_sun/arcsec^2]
        return self._sigma_crit_arcsec

    def phys2arcsec_lens(self, phys):
        """Convert physical Mpc into arc seconds.

        :param phys: physical distance [Mpc]
        :return: angular diameter [arcsec]
        """
        return phys / self.dd / const.arcsec

    def arcsec2phys_lens(self, arcsec):
        """Convert angular to physical quantities for lens plane.

        :param arcsec: angular size at lens plane [arcsec]
        :return: physical size at lens plane [Mpc]
        """
        return arcsec * const.arcsec * self.dd

    def arcsec2phys_source(self, arcsec):
        """Convert angular to physical quantities for source plane.

        :param arcsec: angular size at source plane [arcsec]
        :return: physical size at source plane [Mpc]
        """
        return arcsec * const.arcsec * self.ds

    def kappa2proj_mass(self, kappa):
        """Convert convergence to projected mass M_sun/Mpc^2.

        :param kappa: lensing convergence
        :return: projected mass [M_sun/Mpc^2]
        """
        return kappa * self.sigma_crit

    def mass_in_theta_E(self, theta_E):
        """Mass within Einstein radius (area * epsilon crit) [M_sun]

        :param theta_E: Einstein radius [arcsec]
        :return: mass within Einstein radius [M_sun]
        """
        mass = self.arcsec2phys_lens(theta_E) ** 2 * np.pi * self.sigma_crit
        return mass

    def mass_in_coin(self, theta_E):
        """

        :param theta_E: Einstein radius [arcsec]
        :return: mass in coin calculated in mean density of the universe
        """
        chi_L = self.background.T_xy(0, self.z_lens)
        chi_S = self.background.T_xy(0, self.z_source)
        return (
            1.0
            / 3
            * np.pi
            * (chi_L * theta_E * const.arcsec) ** 2
            * chi_S
            * self.background.rho_crit
        )  # [M_sun/Mpc**3]

    def time_delay_units(self, fermat_pot, kappa_ext=0):
        """

        :param fermat_pot: in units of arcsec^2 (e.g. Fermat potential)
        :param kappa_ext: unit-less external shear not accounted for in the Fermat potential
        :return: time delay in days
        """
        D_dt = self.ddt * (1.0 - kappa_ext) * const.Mpc  # eqn 7 in Suyu et al.
        return (
            D_dt / const.c * fermat_pot / const.day_s * const.arcsec**2
        )  # * self.arcsec2phys_lens(1.)**2

    def time_delay2fermat_pot(self, dt):
        """

        :param dt: time delay in units of days
        :return: Fermat potential in units arcsec**2 for a given cosmology
        """
        D_dt = self.ddt * const.Mpc
        return dt * const.c * const.day_s / D_dt / const.arcsec**2

    def nfw_angle2physical(self, Rs_angle, alpha_Rs):
        """Converts the angular parameters into the physical ones for an NFW profile.

        :param alpha_Rs: observed bending angle at the scale radius in units of arcsec
        :param Rs_angle: scale radius in units of arcsec
        :return: rho0 [Msun/Mpc^3], Rs [Mpc], c, r200 [Mpc], M200 [Msun]
        """
        Rs = Rs_angle * const.arcsec * self.dd
        theta_scaled = alpha_Rs * self.sigma_crit * self.dd * const.arcsec
        rho0 = theta_scaled / (4 * Rs**2 * (1 + np.log(1.0 / 2.0)))
        rho0_com = rho0 / self.h**2
        c = self.nfw_param.c_rho0(rho0_com, self.z_lens)
        r200 = c * Rs
        M200 = self.nfw_param.M_r200(r200 * self.h, self.z_lens) / self.h
        return rho0, Rs, c, r200, M200

    def nfw_physical2angle(self, M, c):
        """Converts the physical mass and concentration parameter of an NFW profile into
        the lensing quantities.

        :param M: mass enclosed 200 rho_crit in units of M_sun (physical units, meaning
            no little h)
        :param c: NFW concentration parameter (r200/r_s)
        :return: Rs_angle (angle at scale radius) (in units of arcsec), alpha_Rs
            (observed bending angle at the scale radius
        """
        rho0, Rs, r200 = self.nfwParam_physical(M, c)
        Rs_angle = Rs / self.dd / const.arcsec  # Rs in arcsec
        alpha_Rs = rho0 * (4 * Rs**2 * (1 + np.log(1.0 / 2.0)))
        return Rs_angle, alpha_Rs / self.sigma_crit / self.dd / const.arcsec

    def nfwParam_physical(self, M, c):
        """Returns the NFW parameters in physical units.

        :param M: physical mass in M_sun in definition m200
        :param c: concentration
        :return: rho0 [Msun/Mpc^3], Rs [Mpc], r200 [Mpc]
        """
        r200 = (
            self.nfw_param.r200_M(M * self.h, self.z_lens) / self.h
        )  # physical radius r200
        rho0 = (
            self.nfw_param.rho0_c(c, self.z_lens) * self.h**2
        )  # physical density in M_sun/Mpc**3
        Rs = r200 / c
        return rho0, Rs, r200

    def nfw_M_theta_r200(self, M):
        """Returns r200 radius in angular units of arc seconds on the sky.

        :param M: physical mass in M_sun
        :return: angle (in arc seconds) of the r200 radius
        """
        r200 = (
            self.nfw_param.r200_M(M * self.h, self.z_lens) / self.h
        )  # physical radius r200
        theta_r200 = r200 / self.dd / const.arcsec
        return theta_r200

    def sis_theta_E2sigma_v(self, theta_E):
        """Converts the lensing Einstein radius into a physical velocity dispersion.

        :param theta_E: Einstein radius (in arcsec)
        :return: velocity dispersion in units (km/s)
        """
        v_sigma_c2 = theta_E * const.arcsec / (4 * np.pi) * self.ds / self.dds
        return np.sqrt(v_sigma_c2) * const.c / 1000

    def sis_sigma_v2theta_E(self, v_sigma):
        """Converts the velocity dispersion into an Einstein radius for a SIS profile.

        :param v_sigma: velocity dispersion (km/s)
        :return: theta_E (arcsec)
        """
        theta_E = (
            4
            * np.pi
            * (v_sigma * 1000.0 / const.c) ** 2
            * self.dds
            / self.ds
            / const.arcsec
        )
        return theta_E

    def hernquist_phys2angular(self, mass, rs):
        """Translates physical mass definitions of the Hernquist profile to the angular
        units used in the Hernquist lens profile of lenstronomy.

        'sigma0' is defined such that the deflection at projected RS leads to alpha =
        2./3 * Rs * sigma0

        :param mass: A spherical overdensity mass in M_sun corresponding to the mass
            definition mdef at redshift z
        :param rs: rs in units of physical Mpc
        :return: sigma0, Rs_angle
        """
        rs_angle = rs / self.dd / const.arcsec  # Rs in arcsec
        rhos = mass / (2 * np.pi) / rs**3  # units of M_sun / Mpc^3
        sigma0 = rhos * rs  # units of M_sun / Mpc^2
        sigma0 /= self.sigma_crit
        return sigma0, rs_angle

    def hernquist_angular2phys(self, sigma0, rs_angle):
        """'sigma0' is defined such that the deflection at projected RS leads to alpha =
        2./3 * Rs * sigma0.

        :param sigma0: convergence normalization
        :param rs_angle: rs in angular units [arcseconds]
        :return: mass [M_sun], rs  [Mpc]
        """
        rs = rs_angle * self.dd * const.arcsec  # units of Mpc
        rhos = sigma0 / rs * self.sigma_crit
        m_tot = 2 * np.pi * rhos * rs**3
        return m_tot, rs

    def uldm_angular2phys(self, kappa_0, theta_c):
        """Converts the anguar parameters entering the LensModel Uldm() (Ultra Light
        Dark Matter) class in physical masses, i.e. the total soliton mass and the mass
        of the particle.

        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcseconds)
        :return: m_eV_log10, M_sol_log10, the log10 of the masses, m in eV and M in
            M_sun
        """
        D_Lens = self.dd * 10**6  # in parsec
        Sigma_c = self.sigma_crit * 10 ** (-12)  # in M_sun / parsec^2
        r_c = theta_c * const.arcsec * D_Lens
        rho0 = 2048 * np.sqrt(0.091) * kappa_0 * Sigma_c / (429 * np.pi * r_c)
        m_log10 = -22 + 0.5 * np.log10(190 / rho0 * (r_c / 100) ** (-4))
        M_log10 = 9 + np.log10(160 * 1.4 / r_c) - 2 * (m_log10 + 22)
        return m_log10, M_log10

    def uldm_mphys2angular(self, m_log10, M_log10):
        """Converts physical ULDM mass in the ones, in angular units, that enter the
        LensModel Uldm() class.

        :param m_log10: exponent of ULDM mass in eV
        :param M_log10: exponent of soliton mass in M_sun
        :return: kappa_0, theta_c, the central convergence and core radius (in
            arcseconds)
        """
        D_Lens = self.dd * 10**6  # in parsec
        Sigma_c = self.sigma_crit * 10 ** (-12)  # in M_sun/parsec^2
        m22 = 10 ** (m_log10 + 22)
        M9 = 10 ** (M_log10 - 9)
        r_c = 160 * 1.4 * m22 ** (-2) * M9 ** (-1)  # core radius in parsec
        rho0 = (
            190 * m22 ** (-2) * (r_c / 100) ** (-4)
        )  # central density in M_sun/parsec^3
        kappa_0 = 429 * np.pi * rho0 * r_c / (2048 * np.sqrt(0.091) * Sigma_c)
        theta_c = r_c / D_Lens / const.arcsec
        return kappa_0, theta_c

    def sersic_m_star2k_eff(self, m_star, R_sersic, n_sersic):
        """Translates a total stellar mass into 'k_eff', the convergence at 'R_sersic'
        (effective radius or half-light radius) for a Sersic profile.

        :param m_star: total stellar mass in physical Msun
        :param R_sersic: half-light radius in arc seconds
        :param n_sersic: Sersic index
        :return: k_eff
        """
        # compute mass integral
        from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil

        sersic_util = SersicUtil()
        norm_integral = sersic_util.total_flux(
            amp=1, R_sersic=R_sersic, n_sersic=n_sersic
        )
        # compute total kappa normalization and re
        k_eff = m_star / self.sigma_crit_angle
        # renormalize
        k_eff /= norm_integral
        return k_eff

    def sersic_k_eff2m_star(self, k_eff, R_sersic, n_sersic):
        """Translates convergence at half-light radius to total integrated physical
        stellar mass for a Sersic profile.

        :param k_eff: lensing convergence at half-light radius
        :param R_sersic: half-light radius in arc seconds
        :param n_sersic: Sersic index
        :return: stellar mass in physical Msun
        """
        from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil

        sersic_util = SersicUtil()
        norm_integral = sersic_util.total_flux(
            amp=1, R_sersic=R_sersic, n_sersic=n_sersic
        )
        m_star = k_eff * self.sigma_crit_angle * norm_integral
        return m_star
