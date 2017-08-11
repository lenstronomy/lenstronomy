__author__ = 'sibirrer'

import math
import numpy as np
from galkin.LOS_dispersion import Velocity_dispersion
from galkin.galkin import Galkin

import lenstronomy.Cosmo.constants as const
from lenstronomy.Cosmo.unit_manager import UnitManager
from lenstronomy.LensAnalysis.lens_analysis import LensAnalysis


class LensProp(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_options, kwargs_data):
        self.z_d = z_lens
        self.z_s = z_source
        self.unitManager = UnitManager(z_lens, z_source)
        self.lens_analysis = LensAnalysis(kwargs_options, kwargs_data)
        self.kwargs_data = kwargs_data
        self.kwargs_options = kwargs_options
        self.dispersion = Velocity_dispersion()

    def time_delays(self, kwargs_lens, kwargs_source, kwargs_else, kappa_ext=0):
        time_delay_arcsec = self.lens_analysis.fermat_potential(kwargs_lens, kwargs_else)
        time_delay = self.unitManager.time_delay_units(time_delay_arcsec, kappa_ext)
        return time_delay

    def rho0_r0_gamma(self, kwargs_lens, kwargs_else, gamma, kappa_ext=0):
        # equation (14) in Suyu+ 2010
        theta_E = self.lens_analysis.effective_einstein_radius(kwargs_lens, kwargs_else)
        return (kappa_ext - 1) * math.gamma(gamma/2)/(np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E**gamma/self.unitManager.arcsec2phys_lens(theta_E) * self.unitManager.cosmoProp.epsilon_crit * const.M_sun/const.Mpc**3  # units kg/m^3

    def v_sigma(self, kwargs_lens, kwargs_lens_light, kwargs_else, r_ani_scaling=1, r_eff=None, r=0.01):
        """
        returns LOL central velocity dispersion in units of km/s
        :return:
        """
        gamma = kwargs_lens['gamma']
        # equation (14) in Suyu+ 2010
        if r_eff is None:
            r_eff = self.lens_analysis.half_light_radius(kwargs_lens_light)
        rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, kwargs_else, gamma)
        r_ani = r_ani_scaling * r_eff
        sigma2_center = self.dispersion.sigma_r2(r, 0.551*r_eff, gamma, rho0_r0_gamma, r_ani)
        return np.sqrt(sigma2_center) * self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000

    def velocity_dispersion(self, kwargs_lens, kwargs_lens_light, kwargs_else, aniso_param=1, r_eff=None, R_slit=0.81, dR_slit=0.1, psf_fwhm=0.7, num_evaluate=100):
        gamma = kwargs_lens[0]['gamma']
        if r_eff is None:
            r_eff = self.lens_analysis.half_light_radius(kwargs_lens_light)
        theta_E = self.lens_analysis.effective_einstein_radius(kwargs_lens, kwargs_else)

        #rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, kwargs_else, gamma)
        if self.dispersion.beta_const is False:
            aniso_param *= r_eff
        sigma2 = self.dispersion.vel_disp(gamma, theta_E, r_eff, aniso_param, R_slit, dR_slit, FWHM=psf_fwhm, num=num_evaluate)
        return np.sqrt(sigma2) * self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000

    def velocity_disperson_new(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, lens_model_bool, light_model_bool, psf_fwhm):
        """

        :param kwargs_lens:
        :param kwargs_lens_light:
        :param kwargs_anisotropy:
        :param kwargs_aperature:
        :return:
        """
        kwargs_cosmo = {'D_d': self.unitManager.D_d, 'D_s': self.unitManager.D_s, 'D_ds': self.unitManager.D_ds}
        mass_profile_list = []
        kwargs_profile = []
        for i, lens_model in enumerate(self.kwargs_options['lens_model_list']):
            if lens_model_bool[i]:
                mass_profile_list.append(lens_model)
                kwargs_profile.append(kwargs_lens[i])

        light_profile_list = []
        kwargs_light = []
        for i, light_model in enumerate(self.kwargs_options['lens_light_model_list']):
            if light_model_bool[i]:
                light_profile_list.append(light_model)
                kwargs_light.append(kwargs_lens_light[i])

        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=kwargs_aperture['aperture_type'],
                        anisotropy_model=kwargs_anisotropy['anisotropy_type'], fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture, num=1000)
        return sigma_v

    def angular_diameter_relations(self, sigma_v_model, sigma_v, kappa_ext, D_dt_model, z_d):
        """

        :return:
        """
        sigma_v2_model = sigma_v_model**2
        Ds_Dds = sigma_v**2/(1-kappa_ext)/(sigma_v2_model*self.unitManager.cosmoProp.dist_LS/self.unitManager.cosmoProp.dist_OS)
        D_d = D_dt_model/(1+z_d)/Ds_Dds/(1-kappa_ext)
        return D_d, Ds_Dds

    def angular_distances(self, sigma_v_measured, time_delay_measured, kappa_ext, sigma_v_modeled, fermat_pot):
        """

        :param sigma_v_measured: velocity dispersion measured [km/s]
        :param time_delay_measured: time delay measured [d]
        :param kappa_ext: external convergence estimated []
        :param sigma_v_modeled: lens model velocity dispersion with default cosmology and without external convergence [km/s]
        :param fermat_pot: fermat potential of lens model, modulo MSD of kappa_ext [arcsec^2]
        :return: D_d and D_d*D_s/D_ds, units in Mpc physical
        """

        Ds_Dds = (sigma_v_measured/sigma_v_modeled)**2/(self.unitManager.cosmoProp.dist_LS/self.unitManager.cosmoProp.dist_OS)/(1-kappa_ext)
        DdDs_Dds = 1./(1+self.z_d)/(1-kappa_ext) * (const.c * time_delay_measured * const.day_s)/(fermat_pot*const.arcsec**2)/const.Mpc
        return Ds_Dds, DdDs_Dds