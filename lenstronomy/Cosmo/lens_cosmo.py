__author__ = 'sibirrer'

#this file contains a class to convert units

import numpy as np
import lenstronomy.Util.constants as const
from lenstronomy.Cosmo.background import Background
from lenstronomy.Cosmo.nfw_param import NFWParam
from astropy.cosmology import FlatLambdaCDM, LambdaCDM


class LensCosmo(object):
    """
    class to manage the physical units and distances present in a single plane lens with fixed input cosmology
    """
    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy.cosmology instance
        """

        self.z_lens = z_lens
        self.z_source = z_source
        self.background = Background(cosmo=cosmo)
        self.nfw_param = NFWParam()

    def a_z(self, z):
        """
        convert redshift into scale factor
        :param z: redshift
        :return: scale factor
        """
        return 1. / (1. + z)

    @property
    def h(self):
        return self.background.cosmo.H(0).value / 100.

    @property
    def D_d(self):
        """

        :return: angular diameter distance to the deflector [Mpc]
        """
        return self.background.D_xy(0, self.z_lens)

    @property
    def D_s(self):
        """

        :return: angular diameter distance to the source [Mpc]
        """
        return self.background.D_xy(0, self.z_source)

    @property
    def D_ds(self):
        """

        :return: angular diameter distance from deflector to source [Mpc]
        """
        return self.background.D_xy(self.z_lens, self.z_source)

    @property
    def D_dt(self):
        """

        :return: time delay distance [Mpc]
        """
        return (1 + self.z_lens) * self.D_d * self.D_s / self.D_ds

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        :return: critical projected mass density
        """
        if not hasattr(self, '_Epsilon_Crit'):
            const_SI = const.c ** 2 / (4 * np.pi * const.G)  #c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._Epsilon_Crit = self.D_s/(self.D_d*self.D_ds) * factor #[M_sun/Mpc^2]
        return self._Epsilon_Crit

    def phys2arcsec_lens(self, phys):
        """
        convert physical Mpc into arc seconds
        :param phys: physical distance [Mpc]
        :return: angular diameter [arcsec]
        """
        return phys / self.D_d/const.arcsec

    def arcsec2phys_lens(self, arcsec):
        """
        convert angular to physical quantities for lens plane
        :param arcsec: angular size at lens plane [arcsec]
        :return: physical size at lens plane [Mpc]
        """
        return arcsec * const.arcsec * self.D_d

    def arcsec2phys_source(self, arcsec):
        """
        convert angular to physical quantities for source plane
        :param arcsec: angular size at source plane [arcsec]
        :return: physical size at source plane [Mpc]
        """
        return arcsec * const.arcsec * self.D_s

    def kappa2proj_mass(self, kappa):
        """
        convert convergence to projected mass M_sun/Mpc^2
        :param kappa: lensing convergence
        :return: projected mass [M_sun/Mpc^2]
        """
        return kappa * self.epsilon_crit

    def mass_in_theta_E(self, theta_E):
        """
        mass within Einstein radius (area * epsilon crit) [M_sun]
        :param theta_E: Einstein radius [arcsec]
        :return: mass within Einstein radius [M_sun]
        """
        mass = self.arcsec2phys_lens(theta_E) ** 2 * np.pi * self.epsilon_crit
        return mass

    def mass_in_coin(self, theta_E):
        """

        :param theta_E: Einstein radius [arcsec]
        :return: mass in coin calculated in mean density of the universe
        """
        chi_L = self.background.T_xy(0, self.z_lens)
        chi_S = self.background.T_xy(0, self.z_source)
        return 1./3 * np.pi * (chi_L * theta_E * const.arcsec) ** 2 * chi_S * self.background.rho_crit  #[M_sun/Mpc**3]

    def time_delay_units(self, fermat_pot, kappa_ext=0):
        """

        :param delay_unitless: in units of arcsec^2 (e.g. Fermat potential)
        :param kappa_ext: unit-less
        :return: time delay in days
        """
        D_dt = self.D_dt / (1. - kappa_ext) * const.Mpc  # eqn 7 in Suyu et al.
        return D_dt / const.c * fermat_pot / const.day_s * const.arcsec ** 2  # * self.arcsec2phys_lens(1.)**2

    def nfw_angle2physical(self, Rs_angle, theta_Rs):
        """
        converts the angular parameters into the physical ones for an NFW profile
        :param theta_Rs: observed bending angle at the scale radius in units of arcsec
        :param Rs: scale radius in units of arcsec
        :return: M200, r200, Rs_physical, c
        """
        Rs = Rs_angle * const.arcsec * self.D_d
        theta_scaled = theta_Rs * self.epsilon_crit * self.D_d * const.arcsec
        rho0 = theta_scaled / (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        rho0_com = rho0 / self.h**2 * self.a_z(self.z_lens)**3
        c = self.nfw_param.c_rho0(rho0_com)
        r200 = c * Rs
        M200 = self.nfw_param.M_r200(r200 * self.h / self.a_z(self.z_lens)) / self.h
        return rho0, Rs, c, r200, M200

    def nfw_physical2angle(self, M, c):
        """
        converts the physical mass and concentration parameter of an NFW profile into the lensing quantities
        :param M: mass enclosed 200 \rho_crit
        :param c: NFW concentration parameter (r200/r_s)
        :return: theta_Rs (observed bending angle at the scale radius, Rs_angle (angle at scale radius) (in units of arcsec)
        """
        rho0, Rs, r200 = self.nfwParam_physical(M, c)
        Rs_angle = Rs / self.D_d / const.arcsec  # Rs in arcsec
        theta_Rs = rho0 * (4 * Rs ** 2 * (1 + np.log(1. / 2.)))
        return Rs_angle,  theta_Rs / self.epsilon_crit / self.D_d / const.arcsec

    def nfwParam_physical(self, M, c):
        """
        returns the NFW parameters in physical units
        :param M: physical mass in M_sun
        :param c: concentration
        :return:
        """
        r200 = self.nfw_param.r200_M(M * self.h) / self.h * self.a_z(self.z_lens)  # physical radius r200
        rho0 = self.nfw_param.rho0_c(c) * self.h**2 / self.a_z(self.z_lens)**3 # physical density in M_sun/Mpc**3
        Rs = r200/c
        return rho0, Rs, r200

    def sis_theta_E2sigma_v(self, theta_E):
        """
        converts the lensing Einstein radius into a physical velocity dispersion
        :param theta_E: Einstein radius (in arcsec)
        :return: velocity dispersion in units (km/s)
        """
        v_sigma_c2 = theta_E * const.arcsec / (4*np.pi) * self.D_s / self.D_ds
        return np.sqrt(v_sigma_c2)*const.c / 1000

    def sis_sigma_v2theta_E(self, v_sigma):
        """
        converts the velocity dispersion into an Einstein radius for a SIS profile
        :param v_sigma: velocity dispersion (km/s)
        :return: theta_E (arcsec)
        """
        theta_E = 4 * np.pi * (v_sigma * 1000./const.c)**2 * self.D_ds / self.D_s / const.arcsec
        return theta_E


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
        return lensCosmo.D_d

    def D_s(self, H_0, Om0, Ode0=None):
        """
        angular diameter to source
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.D_s

    def D_ds(self, H_0, Om0, Ode0=None):
        """
        angular diameter from deflector to source
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.D_ds

    def D_dt(self, H_0, Om0, Ode0=None):
        """
        time delay distance
        :param H_0: Hubble parameter [km/s/Mpc]
        :param Om0: normalized matter density at present time
        :return: float [Mpc]
        """
        lensCosmo = self._get_cosom(H_0, Om0, Ode0)
        return lensCosmo.D_dt
