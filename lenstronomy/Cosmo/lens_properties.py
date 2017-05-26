__author__ = 'sibirrer'

import copy
import math

import astrofunc.util as util
import numpy as np
from galkin.LOS_dispersion import Velocity_dispersion

import lenstronomy.Cosmo.constants as const
from lenstronomy.Cosmo.unit_manager import UnitManager
from lenstronomy.Trash.make_image import MakeImage


class LensProp(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_options, kwargs_data):
        self.z_d = z_lens
        self.z_s = z_source
        self.unitManager = UnitManager(z_lens, z_source)
        #self.timeDelaySampling = TimeDelaySampling()
        self.makeImage = MakeImage(kwargs_options, kwargs_data)
        self.kwargs_data = kwargs_data

#        self.kwargs_data = kwargs_data
#        self.kwargs_options = kwargs_options
        self.dispersion = Velocity_dispersion()

    def time_delays(self, kwargs_lens, kwargs_source, kwargs_else, kappa_ext=0):
        time_delay_arcsec = self.makeImage.fermat_potential(kwargs_lens, kwargs_source, kwargs_else)
        time_delay = self.unitManager.time_delay_units(time_delay_arcsec, kappa_ext)
        return time_delay

    def magnification_ratios(self, kwargs_lens, kwargs_else, param):
        x_pos, y_pos, mag_model = self.makeImage.get_magnification_model(kwargs_lens, kwargs_else)
        mag_data = self.makeImage.get_image_amplitudes(param, kwargs_else)
        return mag_data, mag_model

    def effective_einstein_radius(self, kwargs_lens_list, n_grid=200, delta_grid=0.05):
        """
        computes the radius with mean convergence=1
        :param kwargs_lens:
        :return:
        """
        kwargs_lens = kwargs_lens_list[0]
        kwargs_lens_copy = kwargs_lens.copy()
        kwargs_lens_copy['center_x'] = 0
        kwargs_lens_copy['center_y'] = 0
        x_grid, y_grid = util.make_grid(n_grid, delta_grid)
        kappa = self.makeImage.LensModel.kappa(x_grid, y_grid, [kwargs_lens_copy], k=0)
        kappa = util.array2image(kappa)
        r_array = np.linspace(0.0001, 2*kwargs_lens['theta_E'], 1000)
        for r in r_array:
            mask = np.array(1 - util.get_mask(0, 0, r, x_grid, y_grid))
            kappa_mean = np.sum(kappa*mask)/np.sum(mask)
            if kappa_mean < 1:
                return r
        return -1

    def rho0_r0_gamma(self, kwargs_lens, gamma, kappa_ext=0):
        # equation (14) in Suyu+ 2010
        theta_E = self.effective_einstein_radius(kwargs_lens)
        return (kappa_ext - 1) * math.gamma(gamma/2)/(np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * theta_E**gamma/self.unitManager.arcsec2phys_lens(theta_E) * self.unitManager.cosmoProp.epsilon_crit * const.M_sun/const.Mpc**3  # units kg/m^3

    def v_sigma(self, kwargs_lens, kwargs_lens_light, kwargs_else, r_ani_scaling=1, r_eff=None, r=0.01):
        """
        returns LOL central velocity dispersion in units of km/s
        :return:
        """
        gamma = kwargs_lens['gamma']
        # equation (14) in Suyu+ 2010
        if r_eff is None:
            r_eff = self.half_light_radius(kwargs_lens_light)
        rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, kwargs_else)
        r_ani = r_ani_scaling * r_eff
        sigma2_center = self.dispersion.sigma_r2(r, 0.551*r_eff, gamma, rho0_r0_gamma, r_ani)
        return np.sqrt(sigma2_center) * self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000

    def velocity_dispersion(self, kwargs_lens, kwargs_lens_light, kwargs_else, aniso_param=1, r_eff=None, R_slit=0.81, dR_slit=0.1, psf_fwhm=0.7, num_evaluate=100):
        gamma = kwargs_lens[0]['gamma']
        if r_eff is None:
            r_eff = self.half_light_radius(kwargs_lens_light)
        rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, gamma)
        if self.dispersion.beta_const is False:
            aniso_param *= r_eff
        sigma2 = self.dispersion.vel_disp(gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM=psf_fwhm, num=num_evaluate)
        return np.sqrt(sigma2) * self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000

    def half_light_radius(self, kwargs_lens_light):
        """
        computes numerically the half-light-radius of the deflector light and the total photon flux
        :param kwargs_lens_light:
        :return:
        """
        kwargs_lens_light_copy = copy.deepcopy(kwargs_lens_light)
        kwargs_lens_light_copy['center_x'] = 0
        kwargs_lens_light_copy['center_y'] = 0
        data = self.kwargs_data['image_data']
        numPix = int(np.sqrt(len(data))*10)
        deltaPix = self.kwargs_data['deltaPix']
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        lens_light = self.makeImage.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light_copy)
        R_h = util.half_light_radius(lens_light, x_grid, y_grid)
        return R_h

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