__author__ = 'sibirrer'

import numpy as np
import lenstronomy.GalKin.velocity_util as vel_util
from lenstronomy.GalKin.cosmo import Cosmo
import lenstronomy.Util.constants as const
import math


class Velocity_dispersion(object):
    """
    class to compute eqn 20 in Suyu+2010 with a monte-carlo process
    """
    def __init__(self, beta_const=False, b_prior=False, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}):
        self.beta_const = beta_const
        self.b_prior = b_prior
        self.cosmo = Cosmo(kwargs_cosmo)

    def vel_disp(self, gamma, theta_E, r_eff, aniso_param, R_slit, dR_slit, FWHM, num=100):
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
        if self.b_prior and self.beta_const:
            aniso_param = self.b_beta(aniso_param)
        sigma_s2_sum = 0
        rho0_r0_gamma = self.rho0_r0_gamma(theta_E, gamma)
        for i in range(0, num):
            sigma_s2_draw = self.vel_disp_one(gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM)
            sigma_s2_sum += sigma_s2_draw
        sigma_s2_average = sigma_s2_sum/num
        return np.sqrt(sigma_s2_average)

    def rho0_r0_gamma(self, theta_E, gamma):
        # equation (14) in Suyu+ 2010
        return -1 * math.gamma(gamma/2)/(np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E**gamma/self.cosmo.arcsec2phys_lens(theta_E) * self.cosmo.epsilon_crit * const.M_sun/const.Mpc**3  # units kg/m^3

    def vel_disp_one(self, gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM):
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
        a = 0.551 * r_eff
        while True:
            r = self.P_r(a)  # draw r
            R, x, y = self.R_r(r)  # draw projected R
            x_, y_ = self.displace_PSF(x, y, FWHM)  # displace via PSF
            bool = self.check_in_slit(x_, y_, R_slit, dR_slit)
            if bool is True:
                break
        sigma_s2 = self.sigma_s2(r, R, aniso_param, a, gamma, rho0_r0_gamma)
        return sigma_s2

    def P_r(self, a):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r

    def R_r(self, r):
        """
        draws a random projection from radius r in 2d and 1d
        :param r: 3d radius
        :return: R, x, y
        """
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        R = np.sqrt(x**2 + y**2)
        return R, x, y

    def displace_PSF(self, x, y, FWHM):
        """

        :param x: x-coord (arc sec)
        :param y: y-coord (arc sec)
        :param FWHM: psf size (arc sec)
        :return: x', y' random displaced according to psf
        """
        sigma = FWHM/(2*np.sqrt(2*np.log(2)))
        sigma_one_direction = sigma
        x_ = x + np.random.normal() * sigma_one_direction
        y_ = y + np.random.normal() * sigma_one_direction
        return x_, y_

    def check_in_slit(self, x, y, R_slit, dR_slit):
        """

        check whether a ray in position (x,y) is captured in the slit with Radius R_slit and width dR_slit

        :param x:
        :param y:
        :param R_slit:
        :param dR_slit:
        :return:
        """
        if abs(x) < R_slit/2. and abs(y) < dR_slit/2.:
            return True
        else:
            return False

    def sigma_s2(self, r, R, aniso_param, a, gamma, rho0_r0_gamma):
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
        if self.beta_const:
            beta = aniso_param
            r_ani = self._ani_beta(r, aniso_param)
        else:
            r_ani = aniso_param
            beta = self._beta_ani(r, r_ani)
        return (1 - beta * R**2/r**2) * self.sigma_r2(r, a, gamma, rho0_r0_gamma, r_ani)

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
        return prefac1 * prefac2 * fac * (self.cosmo.arcsec2phys_lens(1.) * const.Mpc / 1000)**2

    def _beta_ani(self, r, r_ani):
        """
        anisotropy parameter beta
        :param r:
        :param r_ani:
        :return:
        """
        return r**2/(r_ani**2 + r**2)

    def _ani_beta(self, r, beta):
        """
        given radius and anisotropy beta, what is the "anisotropy radius"
        :param r:
        :param beta:
        :return:
        """
        if beta > 1:
            raise ValueError("Value of beta = %s not valid!" % beta)
        return np.sqrt(r**2*(1./beta-1))

    def b_beta(self, b):
        """

        :param b: 1 - 1/b = beta
        :return:
        """
        assert(b > 0)
        return 1. - 1. / b
