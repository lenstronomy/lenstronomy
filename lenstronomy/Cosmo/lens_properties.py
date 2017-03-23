__author__ = 'sibirrer'

import numpy as np
from scipy import integrate
import math

from lenstronomy.Cosmo.unit_manager import UnitManager
from lenstronomy.Cosmo.time_delay import TimeDelay
from lenstronomy.Cosmo.time_delay_sampling import TimeDelaySampling
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.MCMC.compare import Compare
from lenstronomy.Cosmo.VelocityDispersion.spectral_apperature import Apperature
from lenstronomy.Cosmo.VelocityDispersion.LOS_dispersion import Velocity_dispersion
import lenstronomy.Cosmo.constants as const
import lenstronomy.util as util

class LensProp(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc for a given lens model
    """

    def __init__(self, LensSystem, kwargs_options, kwargs_data):
        self.unitManager = UnitManager(LensSystem.z_lens, LensSystem.z_source)
        self.timeDelay = TimeDelay(LensSystem.z_lens, LensSystem.z_source)
        self.timeDelaySampling = TimeDelaySampling()
        self.makeImage = MakeImage(kwargs_options, kwargs_data)
        self.compare = Compare(kwargs_options)

        self.kwargs_data = kwargs_data
        self.kwargs_options = kwargs_options
        self.apperature = Apperature()
        self.dispersion = Velocity_dispersion()

    def time_delays(self, kwargs_lens, kwargs_source, kwargs_else):
        time_delay_arcsec = self.makeImage.get_time_delay(kwargs_lens, kwargs_source, kwargs_else)
        kappa_external = kwargs_else.get('kappa_external', 0)
        time_delay = self.timeDelay.time_delay_units(time_delay_arcsec, kappa_external)
        return time_delay

    def logL_delay_dist(self, kwargs_lens, kwargs_source, kwargs_else):
        delay_arcsec = self.makeImage.get_time_delay(kwargs_lens, kwargs_source, kwargs_else)
        D_dt_model = kwargs_else['delay_dist']
        delay_days = self.timeDelaySampling.days_D_model(delay_arcsec, D_dt_model)
        logL = self.compare.delays(delay_days, self.kwargs_data['time_delays'], self.kwargs_data['time_delays_errors'])
        return logL

    def magnification_ratios(self, kwargs_lens, kwargs_else, param):
        x_pos, y_pos, mag_model = self.makeImage.get_magnification_model(kwargs_lens, kwargs_else)
        mag_data = self.makeImage.get_image_amplitudes(param, kwargs_else)
        return mag_data, mag_model

    def effective_einstein_radius(self, kwargs_lens, n_grid=100, delta_grid=0.05):
        """
        computes the radius with mean convergence=1
        :param kwargs_lens:
        :return:
        """
        x_grid, y_grid = util.make_grid(n_grid, delta_grid)
        kappa = self.makeImage.LensModel.kappa(x_grid, y_grid, **kwargs_lens)
        kappa = util.array2image(kappa)
        r_array = np.linspace(0, 2*kwargs_lens['phi_E'], 1000)
        for r in r_array:
            mask = 1 - util.get_mask(kwargs_lens['center_x'], kwargs_lens['center_x'], r, x_grid, y_grid)
            kappa_mean = np.sum(kappa*mask)/np.sum(mask)
            if kappa_mean < 1:
                return r
        return -1

    def rho0_r0_gamma(self, kwargs_lens, kwargs_else):
        # equation (14) in Suyu+ 2010
        gamma = kwargs_lens['gamma']
        kappa_external = kwargs_else.get('kappa_external', 0.)
        phi_E = self.effective_einstein_radius(kwargs_lens)
        return (kappa_external - 1) * math.gamma(gamma/2)/(np.sqrt(np.pi)*math.gamma((gamma-3)/2.)) * phi_E**gamma/self.unitManager.arcsec2phys_lens(phi_E) * self.unitManager.cosmoProp.epsilon_crit * const.M_sun/const.Mpc**3  # units kg/m^3

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
        gamma = kwargs_lens['gamma']
        if r_eff is None:
            r_eff = self.half_light_radius(kwargs_lens_light)
        rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, kwargs_else)
        if self.dispersion.beta_const is False:
            aniso_param *= r_eff
        sigma2 = self.dispersion.vel_disp(gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM=psf_fwhm, num=num_evaluate)
        return np.sqrt(sigma2) * self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000

    def velocity_dispersion_one(self, kwargs_lens, kwargs_lens_light, kwargs_else, aniso_param=1, r_eff=None, R_slit=0.81, dR_slit=0.1, psf_fwhm=0.7):
        gamma = kwargs_lens['gamma']
        if r_eff is None:
            r_eff = self.half_light_radius(kwargs_lens_light)
        rho0_r0_gamma = self.rho0_r0_gamma(kwargs_lens, kwargs_else)
        if self.dispersion.beta_const is False:
            aniso_param *= r_eff
        sigma2 = self.dispersion.vel_disp_one(gamma, rho0_r0_gamma, r_eff, aniso_param, R_slit, dR_slit, FWHM=psf_fwhm)
        return sigma2 * (self.unitManager.arcsec2phys_lens(1.) * const.Mpc/1000)**2

    def angular_diameter_relations(self, sigma_v_model, sigma_v, kappa_ext, D_dt_model, z_d):
        """

        :return:
        """
        sigma_v2_model = sigma_v_model**2
        Ds_Dds = sigma_v**2/(1-kappa_ext)/(sigma_v2_model*self.unitManager.cosmoProp.dist_LS/self.unitManager.cosmoProp.dist_OS)
        D_d = D_dt_model/(1+z_d)/Ds_Dds/(1-kappa_ext)
        return D_d, Ds_Dds

    def print_prop(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, param):
        """
        prints all the statements
        :param LensSystem:
        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_else:
        :return:
        """
        phi_E = kwargs_lens['phi_E']
        print('Einstein radius: ', phi_E)
        print('Mass within Einstein radius (log_10): ', np.log10(self.unitManager.mass_in_phi_E(phi_E)))
        print('Point source positions at ra-axes:', kwargs_else['ra_pos'])
        print('Point source positions at dec-axes:', kwargs_else['dec_pos'])
        time_delay = self.time_delays(kwargs_lens, kwargs_source, kwargs_else)
        print('Time delay distance from Planck Cosmology: [Mpc]', self.unitManager.cosmoProp.D_dt_model)
        print('Angular diameter distances D_d, D_s, D_ds:', self.unitManager.cosmoProp.dist_OL, self.unitManager.cosmoProp.dist_OS, self.unitManager.cosmoProp.dist_LS)
        print('Time delays in days:', time_delay)
        print('Relative time delays: (1-2)', time_delay[0] - time_delay[1])
        print('Relative time delays: (2-3)', time_delay[1] - time_delay[2])
        print('Relative time delays: (3-4)', time_delay[2] - time_delay[3])
        print('Relative time delays: (4-1)', time_delay[3] - time_delay[0])
        print('Relative time delays: (1-3)', time_delay[0] - time_delay[2])
        print('Relative time delays: (2-4)', time_delay[1] - time_delay[3])
        mag_data, mag_model = self.magnification_ratios(kwargs_lens, kwargs_else, param)
        print('Measured brightness of point sources:', mag_data)
        print('Predicted magnifications of point sources:', mag_model)
        mag_norm = mag_data * abs(mag_model[0])/mag_data[0]
        print('Normalized predicted magnifications of point sources', mag_norm)
        if 'coupling' in kwargs_lens:
            print('Mass in dipole in log10: ', np.log10(self.unitManager.mass_in_dipole(kwargs_lens['coupling'], kwargs_lens['phi_E'])))
            r_angle = np.sqrt((kwargs_lens['center_x'] - kwargs_lens['center_x_spp'])**2 + (kwargs_lens['center_y'] - kwargs_lens['center_y_spp'])**2)
            coupling = self.unitManager.estimated_dipole(kwargs_lens['phi_E'], kwargs_lens['phi_E_spp'], r_angle)
            print('Estimated coupling of dipole:', coupling)
        mean_sigma, center_sigma = self.velocity_dispersion(kwargs_lens, kwargs_lens_light, kwargs_else, r_ani_scaling=1, r_eff=None, R_slit=0.81, dR_slit=0.1, psf_fwhm=0.7, num_evaluate=11)
        print('LOS Velocity Dispersion estimate [km/s]: ', mean_sigma, center_sigma)

