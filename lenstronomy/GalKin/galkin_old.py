import numpy as np

import lenstronomy.GalKin.velocity_util as util
from lenstronomy.GalKin.light_profile import LightProfile_old
from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.GalKin.jeans_equation import Jeans_solver


class GalKin_analytic(object):
    """
    master class for all computations
    """
    def __init__(self, aperture='slit', mass_profile='power_law', light_profile='Hernquist', anisotropy_type='r_ani', psf_fwhm=0.7, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}):
        """
        initializes the observation condition and masks
        :param aperture_type: string
        :param psf_fwhm: float
        """
        self._mass_profile = mass_profile
        self._fwhm = psf_fwhm
        self._kwargs_cosmo = kwargs_cosmo
        self.lightProfile = LightProfile_old(light_profile)
        self.aperture = Aperture(aperture)
        self.anisotropy = Anisotropy(anisotropy_type)
        self.FWHM = psf_fwhm
        self.jeans_solver = Jeans_solver(kwargs_cosmo, mass_profile, light_profile, anisotropy_type)

    def vel_disp(self, kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy, num=1000):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)
        :param gamma:
        :param phi_E:
        :param r_eff:
        :param r_ani:
        :param R_slit:
        :param FWHM:
        :return:
        """
        sigma_s2_sum = 0
        for i in range(0, num):
            sigma_s2_draw = self._vel_disp_one(kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy)
            sigma_s2_sum += sigma_s2_draw
        sigma_s2_average = sigma_s2_sum/num
        return np.sqrt(sigma_s2_average)

    def _vel_disp_one(self, kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy):
        """
        computes one realisation of the velocity dispersion realized in the slit
        :param gamma:
        :param rho0_r0_gamma:
        :param r_eff:
        :param r_ani:
        :param R_slit:
        :param dR_slit:
        :param FWHM:
        :return:
        """

        while True:
            r = self.lightProfile.draw_light(kwargs_light)  # draw r
            R, x, y = util.R_r(r)  # draw projected R
            x_, y_ = util.displace_PSF(x, y, self.FWHM)  # displace via PSF
            bool = self.aperture.aperture_select(x_, y_, kwargs_aperture)
            if bool is True:
                break
        sigma_s2 = self.sigma_s2(r, R, kwargs_profile, kwargs_anisotropy, kwargs_light)
        return sigma_s2

    def sigma_s2(self, r, R, kwargs_profile, kwargs_anisotropy, kwargs_light):
        """
        projected velocity dispersion
        :param r:
        :param R:
        :param r_ani:
        :param a:
        :param gamma:
        :param phi_E:
        :return:
        """
        beta = self.anisotropy.beta_r(r, kwargs_anisotropy)
        return (1 - beta * R**2/r**2) * self.sigma_r2(r, kwargs_profile, kwargs_anisotropy, kwargs_light)

    def sigma_r2(self, r, kwargs_profile, kwargs_anisotropy, kwargs_light):
        """
        computes radial velocity dispersion at radius r (solving the Jeans equation
        :param r:
        :return:
        """
        return self.jeans_solver.sigma_r2(r, kwargs_profile, kwargs_anisotropy, kwargs_light)
