__author__ = 'sibirrer'


import numpy as np
from astropy.cosmology import default_cosmology

from lenstronomy.Util import class_creator
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.kinematics_api import KinematicAPI
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis


class TDCosmography(object):
    """
    class equipped to perform a cosmographic analysis from a lens model with added measurements of time delays and
    kinematics.
    This class does not require any cosmological knowledge and can return angular diameter distance estimates
    self-consistently integrating the kinematics routines and time delay estimates in the lens modeling.
    This description follows Birrer et al. 2016, 2019.


    """
    def __init__(self, z_lens, z_source, kwargs_model, cosmo_fiducial=None, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None):

        if cosmo_fiducial is None:
            cosmo_fiducial = default_cosmology.get()
        self._cosmo_fiducial = cosmo_fiducial
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo_fiducial)
        self._kinematic_api = KinematicAPI(z_lens, z_source, kwargs_model, cosmo=self._cosmo_fiducial)
        self.LensModel, self.SourceModel, self.LensLightModel, self.PointSource, extinction_class = class_creator.create_class_instances(all_models=True, **kwargs_model)
        self._lensLightProfile = LightProfileAnalysis(light_model=self.LensLightModel)
        self._lensMassProfile = LensProfileAnalysis(lens_model=self.LensModel)
        self._lens_model_kinematics_bool = lens_model_kinematics_bool
        self._light_model_kinematics_bool = light_model_kinematics_bool

    def time_delays(self, kwargs_lens, kwargs_ps, kappa_ext=0):
        """
        predicts the time delays of the image positions given the fiducial cosmology

        :param kwargs_lens: lens model parameters
        :param kwargs_ps: point source parameters
        :param kappa_ext: external convergence (optional)
        :return: time delays at image positions for the fixed cosmology
        """
        fermat_pot = self.fermat_potential(kwargs_lens, kwargs_ps)
        time_delay = self._lens_cosmo.time_delay_units(fermat_pot, kappa_ext)
        return time_delay

    def fermat_potential(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source keyword argument list
        :return: Fermat potential of all the image positions in the first point source list entry
        """
        ra_pos, dec_pos = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        ra_pos = ra_pos[0]
        dec_pos = dec_pos[0]
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens)
        sigma_source = np.sqrt(np.var(ra_source) + np.var(dec_source))
        if sigma_source > 0.001:
            Warning('Source position computed from the different image positions do not trace back to the same position! '
                    'The error is %s mas and may be larger than what is required for an accurate relative time delay estimate!'
                    'See e.g. Birrer & Treu 2019.' % sigma_source * 1000)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        fermat_pot = self.LensModel.fermat_potential(ra_pos, dec_pos, ra_source, dec_source, kwargs_lens)
        return fermat_pot

    def angular_diameter_relations(self, sigma_v_model, sigma_v, kappa_ext, D_dt_model):
        """

        :return:
        """
        sigma_v2_model = sigma_v_model**2
        Ds_Dds = sigma_v**2/(1-kappa_ext)/(sigma_v2_model * self._lens_cosmo.D_ds / self._lens_cosmo.D_s)
        D_d = D_dt_model/(1+self._lens_cosmo.z_lens)/Ds_Dds/(1-kappa_ext)
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

        Ds_Dds = (sigma_v_measured/float(sigma_v_modeled)) ** 2 / (self._lens_cosmo.D_ds / self._lens_cosmo.D_s) / (1. - kappa_ext)
        DdDs_Dds = 1./(1+self._lens_cosmo.z_lens)/(1. - kappa_ext) * (const.c * time_delay_measured * const.day_s)/(fermat_pot*const.arcsec**2)/const.Mpc
        return Ds_Dds, DdDs_Dds

    def kinematics_dimension_less(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, anisotropy_model, r_eff=None,
                                  theta_E=None, gamma=None, analytic_kinematics=False, Hernquist_approx=False, MGE_light=False,
                                  MGE_mass=False):
        """
        \sigma^2 = D_d/D_ds * c^2 *J(kwargs_lens, kwargs_light, anisotropy) (Equation 4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019)
        J() is a dimensionless and cosmological independent quantity only depending on angular units
        This function returns J given the lens and light parameters and the anisotropy choice without an external mass sheet correction.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param r_eff: projected half-light radius of the stellar light associated with the deflector galaxy, optional,
         if set to None will be computed in this function with default settings that may not be accurate.
        :param analytic_kinematics: boolean, if True, used the analytic JAM modeling for a power-law profile on top of a Hernquist light profile
         ATTENTION: This may not be accurate for your specific problem!
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """

        if analytic_kinematics is True:
            if r_eff is None:
                r_eff = self._lensLightProfile.half_light_radius(kwargs_lens_light, grid_spacing=0.05, grid_num=200,
                                                                 center_x=None, center_y=None,
                                                                 model_bool_list=self._light_model_kinematics_bool)
            if theta_E is None:
                theta_E = self._lensMassProfile.effective_einstein_radius(kwargs_lens, center_x=None, center_y=None,
                                                              model_bool_list=self._lens_model_kinematics_bool, grid_num=200, grid_spacing=0.05,
                                                              get_precision=False, verbose=True)
            if gamma is None:
                gamma = self._lensMassProfile.profile_slope(kwargs_lens, theta_E, center_x=None, center_y=None,
                                                            model_list_bool=self._lens_model_kinematics_bool,
                                                            num_points=10)
            r_ani = kwargs_anisotropy.get('r_ani')
            num_evaluate = self._kwargs_numerics_kin.get('sampling_number', 1000)
            sigma_v = self._kinematic_api.velocity_dispersion(theta_E, gamma, r_eff, self._kwargs_aperture_kin,
                                                              self._kwargs_psf_kin, r_ani=r_ani,
                                                              num_evaluate=num_evaluate, kappa_ext=0)
        else:
            sigma_v = self._kinematic_api.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light,
                                                                        kwargs_anisotropy=kwargs_anisotropy,
                                                                        kwargs_aperture=self._kwargs_aperture_kin,
                                                                        kwargs_psf=self._kwargs_psf_kin,
                                                                        anisotropy_model=anisotropy_model,
                                                                        r_eff=r_eff, theta_E=theta_E,
                                                                        kwargs_numerics=self._kwargs_numerics_kin,
                                                                        MGE_light=MGE_light, MGE_mass=MGE_mass,
                                                                        lens_model_kinematics_bool=self._lens_model_kinematics_bool,
                                                                        light_model_kinematics_bool=self._light_model_kinematics_bool,
                                                                        Hernquist_approx=Hernquist_approx, kappa_ext=0)
        J = sigma_v**2 * self._lens_cosmo.D_ds / self._lens_cosmo.D_s / const.c**2
        return J

    def kinematic_observation_settings(self, kwargs_aperture, kwargs_seeing, kwargs_numerics):
        """

        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics: numerical settings for the integrated line-of-sight velocity dispersion
        :return: None
        """
        self._kwargs_aperture_kin = kwargs_aperture
        self._kwargs_numerics_kin = kwargs_numerics
        self._kwargs_psf_kin = kwargs_seeing
