__author__ = 'sibirrer'

import numpy as np
import lenstronomy.GalKin.velocity_util as vel_util
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.GalKin.observation import GalkinObservation
import lenstronomy.Util.constants as const
import math


class AnalyticKinematics(GalkinObservation, Anisotropy):
    """
    class to compute eqn 20 in Suyu+2010 with a Monte-Carlo from rendering from the
    light profile distribution and displacing them with a Gaussian seeing convolution

    This class assumes spherical symmetry in light and mass distribution and
        - a Hernquist light profile (parameterised by the half-light radius)
        - a power-law mass profile (parameterized by the Einstein radius and logarithmic slop)

    The analytic equations for the kinematics in this approximation are presented e.g. in Suyu et al. 2010 and
    the spectral rendering approach to compute the seeing convolved slit measurement is presented in Birrer et al. 2016.
    The stellar anisotropy is parameterised based on Osipkov 1979; Merritt 1985.

    Units
    -----
    all units are meant to be in angular arc seconds. The physical units are fold in through the angular diameter
    distances

    """
    def __init__(self, kwargs_aperture, kwargs_psf, kwargs_cosmo):
        """

        :param kwargs_aperture:
        :param kwargs_psf:
        :param kwargs_cosmo:
        """

        self._cosmo = Cosmo(**kwargs_cosmo)
        GalkinObservation.__init__(self, kwargs_psf=kwargs_psf, kwargs_aperture=kwargs_aperture)
        Anisotropy.__init__(self, anisotropy_type='OsipkovMerritt')

    def vel_disp(self, gamma, theta_E, r_eff, r_ani, rendering_number=1000):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)

        :param gamma: power-law slope of the mass profile (isothermal = 2)
        :param theta_E: Einstein radius of the lens (in arcseconds)
        :param r_eff: half light radius of the Hernquist profile (or as an approximation of any other profile to be described as a Hernquist profile
        :param r_ani: anisotropy radius
        :param rendering_number: number of spectral renderings drawn from the light distribution that go through the
            slit of the observations

        :return: LOS integrated velocity dispersion in units [km/s]
        """
        sigma_s2_sum = 0
        rho0_r0_gamma = self._rho0_r0_gamma(theta_E, gamma)
        for i in range(0, rendering_number):
            sigma_s2_draw = self.vel_disp_one(gamma, rho0_r0_gamma, r_eff, r_ani)
            sigma_s2_sum += sigma_s2_draw
        sigma_s2_average = sigma_s2_sum / rendering_number
        return np.sqrt(sigma_s2_average) / 1000

    def _rho0_r0_gamma(self, theta_E, gamma):
        # equation (14) in Suyu+ 2010
        return -1 * math.gamma(gamma/2) / (np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E ** gamma / \
               self._cosmo.arcsec2phys_lens(theta_E) * self._cosmo.epsilon_crit * const.M_sun / const.Mpc ** 3

    def vel_disp_one(self, gamma, rho0_r0_gamma, r_eff, r_ani):
        """
        computes one realisation of the velocity dispersion realized in the slit

        :param gamma: power-law slope of the mass profile (isothermal = 2)
        :param rho0_r0_gamma: combination of Einstein radius and power-law slope as equation (14) in Suyu+ 2010
        :param r_eff: half light radius of the Hernquist profile (or as an approximation of any other profile to be described as a Hernquist profile
        :param r_ani: anisotropy radius
        :return: projected velocity dispersion of a single drawn position in the potential [km/s]
        """
        a = 0.551 * r_eff
        while True:
            r, R, x, y = self.draw_light({'a': a, 'r_eff': r_eff})
            x_, y_ = self.displace_psf(x, y)
            bool, _ = self.aperture_select(x_, y_)
            if bool is True:
                break
        sigma_s2 = self._sigma_s2(r, R, r_ani, a, gamma, rho0_r0_gamma)
        return np.asarray(sigma_s2, dtype=float)

    @staticmethod
    def draw_hernquist(a):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r

    def draw_light(self, kwargs_light):
        """

        :param kwargs_light: keyword argument (list) of the light model
        :return: 3d radius (if possible), 2d projected radius, x-projected coordinate, y-projected coordinate
        """
        if 'a' not in kwargs_light:
            kwargs_light['a'] = 0.551 * kwargs_light['r_eff']
        a = kwargs_light['a']
        r = self.draw_hernquist(a)
        R, x, y = vel_util.project2d_random(r)
        return r, R, x, y

    def _sigma_s2(self, r, R, r_ani, a, gamma, rho0_r0_gamma):
        """
        projected velocity dispersion
        :param r: 3d radius of the light tracer particle
        :param R: 2d projected radius of the light tracer particle
        :param r_ani: anisotropy radius
        :param a: scale of the Hernquist light profile
        :param gamma: power-law slope of the mass profile
        :param rho0_r0_gamma: combination of Einstein radius and power-law slope as equation (14) in Suyu+ 2010
        :return: projected velocity dispersion
        """
        beta = self.beta_r(r, **{'r_ani': r_ani})
        return (1 - beta * R**2/r**2) * self.sigma_r2(r, a, gamma, rho0_r0_gamma, r_ani)

    def sigma_s2(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified projected radius

        :param r: 3d radius (not needed for this calculation)
        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: line-of-sight projected velocity dispersion at projected radius R from 3d radius r
        """
        if 'a' not in kwargs_light:
            kwargs_light['a'] = 0.551 * kwargs_light['r_eff']
        if 'rho0_r0_gamma' not in kwargs_mass:
            kwargs_mass['rho0_r0_gamma'] = self._rho0_r0_gamma(kwargs_mass['theta_E'], kwargs_mass['gamma'])
        a = kwargs_light['a']
        gamma = kwargs_mass['gamma']
        rho0_r0_gamma = kwargs_mass['rho0_r0_gamma']
        r_ani = kwargs_anisotropy['r_ani']
        return self._sigma_s2(r, R, r_ani, a, gamma, rho0_r0_gamma)

    def sigma_r2(self, r, a, gamma, rho0_r0_gamma, r_ani):
        """
        equation (19) in Suyu+ 2010
        """
        # first term
        prefac1 = 4*np.pi * const.G * a**(-gamma) * rho0_r0_gamma / (3-gamma)
        prefac2 = r * (r + a)**3/(r**2 + r_ani**2)
        hyp1 = vel_util.hyp_2F1(a=2+gamma, b=gamma, c=3+gamma, z=1./(1+r/a))
        hyp2 = vel_util.hyp_2F1(a=3, b=gamma, c=1+gamma, z=-a/r)
        fac = r_ani**2/a**2 * hyp1 / ((2+gamma) * (r/a + 1)**(2+gamma)) + hyp2 / (gamma*(r/a)**gamma)
        return prefac1 * prefac2 * fac * (const.arcsec * self._cosmo.dd * const.Mpc) ** 2

    def delete_cache(self):
        """
        deletes temporary cache tight to a specific model

        :return:
        """
        pass
