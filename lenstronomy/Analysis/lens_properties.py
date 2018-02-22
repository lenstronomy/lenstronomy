__author__ = 'sibirrer'

import numpy as np
from lenstronomy.GalKin.LOS_dispersion import Velocity_dispersion
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.multi_gauss_expansion as mge
import lenstronomy.Util.constants as const


class LensProp(object):
    """
    this class contains routines to compute time delays, magnification ratios, line of sight velocity dispersions etc for a given lens model
    """

    def __init__(self, z_lens, z_source, kwargs_model, cosmo=None):
        self.z_d = z_lens
        self.z_s = z_source
        self.lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        self.lens_analysis = LensAnalysis(kwargs_model)
        self.lens_model = LensModelExtensions(lens_model_list=kwargs_model['lens_model_list'])
        self.kwargs_options = kwargs_model
        kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}
        self.dispersion = Velocity_dispersion(kwargs_cosmo=kwargs_cosmo)

    def time_delays(self, kwargs_lens, kwargs_ps, kappa_ext=0):
        fermat_pot = self.lens_analysis.fermat_potential(kwargs_lens, kwargs_ps)
        time_delay = self.lensCosmo.time_delay_units(fermat_pot, kappa_ext)
        return time_delay

    def velocity_dispersion(self, kwargs_lens, kwargs_lens_light, aniso_param=1, r_eff=None, R_slit=0.81, dR_slit=0.1, psf_fwhm=0.7, num_evaluate=100):
        gamma = kwargs_lens[0]['gamma']
        if r_eff is None:
            r_eff = self.lens_analysis.half_light_radius_lens(kwargs_lens_light)
        theta_E = kwargs_lens[0]['theta_E']
        if self.dispersion.beta_const is False:
            aniso_param *= r_eff
        sigma2 = self.dispersion.vel_disp(gamma, theta_E, r_eff, aniso_param, R_slit, dR_slit, FWHM=psf_fwhm, num=num_evaluate)
        return sigma2

    def velocity_disperson_numerical(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, psf_fwhm,
                                     aperture_type, anisotropy_model, r_eff=1., kwargs_numerics={}, MGE_light=False, MGE_mass=False):
        """

        :param kwargs_lens:
        :param kwargs_lens_light:
        :param kwargs_anisotropy:
        :param kwargs_aperature:
        :return:
        """
        kwargs_cosmo = {'D_d': self.lensCosmo.D_d, 'D_s': self.lensCosmo.D_s, 'D_ds': self.lensCosmo.D_ds}
        mass_profile_list = []
        kwargs_profile = []
        lens_model_internal_bool = self.kwargs_options.get('lens_model_deflector_bool', [True] * len(kwargs_lens))
        for i, lens_model in enumerate(self.kwargs_options['lens_model_list']):
            if lens_model_internal_bool[i]:
                mass_profile_list.append(lens_model)
                kwargs_lens_i = {k: v for k, v in kwargs_lens[i].items() if not k in ['center_x', 'center_y']}
                kwargs_profile.append(kwargs_lens_i)

        if MGE_mass is True:
            massModel = LensModelExtensions(lens_model_list=mass_profile_list)
            theta_E = massModel.effective_einstein_radius(kwargs_lens)
            r_array = np.logspace(-4, 2, 200) * theta_E
            mass_r = massModel.kappa(r_array, 0, kwargs_profile)
            amps, sigmas, norm = mge.mge_1d(r_array, mass_r, N=20)
            mass_profile_list = ['MULTI_GAUSSIAN_KAPPA']
            kwargs_profile = [{'amp': amps, 'sigma': sigmas}]

        light_profile_list = []
        kwargs_light = []
        lens_light_model_internal_bool = self.kwargs_options.get('light_model_deflector_bool', [True] * len(kwargs_lens_light))
        for i, light_model in enumerate(self.kwargs_options['lens_light_model_list']):
            if lens_light_model_internal_bool[i]:
                light_profile_list.append(light_model)
                kwargs_Lens_light_i = {k: v for k, v in kwargs_lens_light[i].items() if not k in ['center_x', 'center_y']}
                if 'q' in kwargs_Lens_light_i:
                    kwargs_Lens_light_i['q'] = 1
                kwargs_light.append(kwargs_Lens_light_i)

        if MGE_light is True:
            lightModel = LightModel(light_profile_list)
            r_array = np.logspace(-3, 2, 200) * r_eff * 2
            flux_r = lightModel.surface_brightness(r_array, 0, kwargs_light)
            amps, sigmas, norm = mge.mge_1d(r_array, flux_r, N=20)
            light_profile_list = ['MULTI_GAUSSIAN']
            kwargs_light = [{'amp': amps, 'sigma': sigmas}]

        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type,
                        anisotropy_model=anisotropy_model, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, kwargs_numerics=kwargs_numerics)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture, r_eff=r_eff)
        return sigma_v

    def angular_diameter_relations(self, sigma_v_model, sigma_v, kappa_ext, D_dt_model, z_d):
        """

        :return:
        """
        sigma_v2_model = sigma_v_model**2
        Ds_Dds = sigma_v**2/(1-kappa_ext)/(sigma_v2_model * self.lensCosmo.D_ds / self.lensCosmo.D_s)
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

        Ds_Dds = (sigma_v_measured/sigma_v_modeled) ** 2 / (self.lensCosmo.D_ds / self.lensCosmo.D_s) / (1 - kappa_ext)
        DdDs_Dds = 1./(1+self.lensCosmo.z_lens)/(1-kappa_ext) * (const.c * time_delay_measured * const.day_s)/(fermat_pot*const.arcsec**2)/const.Mpc
        return Ds_Dds, DdDs_Dds