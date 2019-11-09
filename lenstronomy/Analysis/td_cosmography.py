__author__ = 'sibirrer'


import numpy as np
from astropy.cosmology import default_cosmology

from lenstronomy.Util import class_creator
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.kinematics_api import KinematicAPI


class TDCosmography(object):
    """
    class equipped to perform a cosmographic analysis from a lens model with added measurements of time delays and
    kinematics.
    This class does not require any cosmological knowledge and can return angular diameter distance estimates
    self-consistently integrating the kinematics routines and time delay estimates in the lens modeling.
    This description follows Birrer et al. 2016, 2019.


    """
    def __init__(self, z_lens, z_source, kwargs_model, cosmo_fiducial=None):

        if cosmo_fiducial is None:
            cosmo_fiducial = default_cosmology.get()
        self._cosmo_fiducial = cosmo_fiducial
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo_fiducial)
        self._kinematic_api = KinematicAPI(z_lens, z_source, kwargs_model, cosmo=self._cosmo_fiducial)
        self.LensModel, self.SourceModel, self.LensLightModel, self.PointSource, extinction_class = class_creator.create_class_instances(all_models=True, **kwargs_model)

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
        ra_pos, dec_pos = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        ra_pos = ra_pos[0]
        dec_pos = dec_pos[0]
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens)
        sigma_source = np.sqrt(np.var(ra_source) + np.var(dec_source))
        if sigma_source > 0.001:
            Warning('Source position computed from the different image positions do not trace back to the same position! '
                    'The error is %s mas and may be larger than what is required for an accurate relative time delay estimate!'
                    'See e.g. Birrer & Treu 2019.' %sigma_source * 1000)
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

    def kinematics_dimension_less(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy):
        """
        \sigma^2 = D_d/D_ds * c^2 *J(kwargs_lens, kwargs_light, anisotropy) (Equation 4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019)
        J() is a dimensionless and cosmological independent quantity only depending on angular units
        This function returns J given the lens and light parameters and the anisotropy choice without an external mass sheet correction.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        sigma_v = self._kinematic_api.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, psf_fwhm,
                                      aperture_type, anisotropy_model, r_eff=None, psf_type='GAUSSIAN', moffat_beta=2.6, kwargs_numerics={}, MGE_light=False,
                                      MGE_mass=False, lens_model_kinematics_bool=None, light_model_kinematics_bool=None,
                                      Hernquist_approx=False, kappa_ext=0)
        if self._kinematic_analytic is True:
            self._kinematic_api.velocity_dispersion(kwargs_lens, r_eff, kwargs_aperture=kwargs_aperture, psf_fwhm=psf_fwhm, aniso_param=1, psf_type='GAUSSIAN',
                            moffat_beta=2.6, num_evaluate=1000, kappa_ext=0)
        J = sigma_v**2 * self._lens_cosmo.D_ds / self._lens_cosmo.D_s / const.c**2
        return J

    def kinematic_observation_settings(self):
        pass