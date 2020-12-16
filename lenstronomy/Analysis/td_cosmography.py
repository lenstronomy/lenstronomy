__author__ = 'sibirrer'


import numpy as np
from astropy.cosmology import default_cosmology
from lenstronomy.Util import class_creator
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.kinematics_api import KinematicsAPI

__all__ = ['TDCosmography']


class TDCosmography(KinematicsAPI):
    """
    class equipped to perform a cosmographic analysis from a lens model with added measurements of time delays and
    kinematics.
    This class does not require any cosmological knowledge and can return angular diameter distance estimates
    self-consistently integrating the kinematics routines and time delay estimates in the lens modeling.
    This description follows Birrer et al. 2016, 2019.


    """
    def __init__(self, z_lens, z_source, kwargs_model, cosmo_fiducial=None, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None, kwargs_seeing={}, kwargs_aperture={}, anisotropy_model=None,
                 multi_observations=False):
        """

        :param z_lens: redshift of deflector
        :param z_source: redshift of source
        :param kwargs_model: model configurations (according to FittingSequence)
        :param cosmo_fiducial: fiducial cosmology used to compute angular diameter distances where required
        :param lens_model_kinematics_bool: (optional) bool list, corresponding to lens models being included into the
         kinematics modeling
        :param light_model_kinematics_bool: (optional) bool list, corresponding to lens light models being included
         into the kinematics modeling
        :param kwargs_seeing: seeing conditions (see observation class in Galkin)
        :param kwargs_aperture: aperture keyword arguments (see aperture class in Galkin)
        :param anisotropy_model: string, anisotropy model type
        :param multi_observations: bool, if True, interprets kwargs_aperture and kwargs_seeing as lists of multiple
         observations
        """

        if cosmo_fiducial is None:
            cosmo_fiducial = default_cosmology.get()
        self._z_lens = z_lens
        self._z_source = z_source
        self._cosmo_fiducial = cosmo_fiducial
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo_fiducial)
        self.LensModel, self.SourceModel, self.LensLightModel, self.PointSource, extinction_class = class_creator.create_class_instances(all_models=True, **kwargs_model)
        super(TDCosmography, self).__init__(z_lens=z_lens, z_source=z_source, kwargs_model=kwargs_model,
                                            cosmo=cosmo_fiducial, lens_model_kinematics_bool=lens_model_kinematics_bool,
                                            light_model_kinematics_bool=light_model_kinematics_bool,
                                            kwargs_seeing=kwargs_seeing, kwargs_aperture=kwargs_aperture,
                                            anisotropy_model=anisotropy_model, multi_observations=multi_observations)

    def time_delays(self, kwargs_lens, kwargs_ps, kappa_ext=0, original_ps_position=False):
        """
        predicts the time delays of the image positions given the fiducial cosmology

        :param kwargs_lens: lens model parameters
        :param kwargs_ps: point source parameters
        :param kappa_ext: external convergence (optional)
        :param original_ps_position: boolean (only applies when first point source model is of type 'LENSED_POSITION'),
         uses the image positions in the model parameters and does not re-compute images (which might be differently ordered) 
         in case of the lens equation solver
        :return: time delays at image positions for the fixed cosmology
        """
        fermat_pot = self.fermat_potential(kwargs_lens, kwargs_ps, original_ps_position=original_ps_position)
        time_delay = self._lens_cosmo.time_delay_units(fermat_pot, kappa_ext)
        return time_delay

    def fermat_potential(self, kwargs_lens, kwargs_ps, original_ps_position=False):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source keyword argument list
        :param original_ps_position: boolean (only applies when first point source model is of type 'LENSED_POSITION'),
         uses the image positions in the model parameters and does not re-compute images (which might be differently ordered) 
         in case of the lens equation solver
        :return: Fermat potential of all the image positions in the first point source list entry
        """
        ra_pos, dec_pos = self.PointSource.image_position(kwargs_ps, kwargs_lens, original_position=original_ps_position)
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
        fermat_pot = self.LensModel.fermat_potential(ra_pos, dec_pos, kwargs_lens, ra_source, dec_source)
        return fermat_pot

    def velocity_dispersion_dimension_less(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=None,
                                           theta_E=None, gamma=None):
        """
        sigma**2 = Dd/Dds * c**2 * J(kwargs_lens, kwargs_light, anisotropy)
        (Equation 4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a dimensionless and
        cosmological independent quantity only depending on angular units. This function returns J given the lens
        and light parameters and the anisotropy choice without an external mass sheet correction.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with the deflector galaxy, optional,
         if set to None will be computed in this function with default settings that may not be accurate.
        :param theta_E: pre-computed Einstein radius (optional)
        :param gamma: pre-computed power-law slope of mass profile
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        sigma_v = self.velocity_dispersion(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light,
                                           kwargs_anisotropy=kwargs_anisotropy, r_eff=r_eff, theta_E=theta_E,
                                           gamma=gamma)
        sigma_v *= 1000 # convert from [km/s] to  [m/s]
        J = sigma_v ** 2 * self._lens_cosmo.dds / self._lens_cosmo.ds / const.c ** 2
        return J

    def velocity_dispersion_map_dimension_less(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=None,
                                           theta_E=None, gamma=None):
        """
        sigma**2 = Dd/Dds * c**2 * J(kwargs_lens, kwargs_light, anisotropy)
        (Equation 4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019) J() is a dimensionless and
        cosmological independent quantity only depending on angular units. This function returns J given the lens
        and light parameters and the anisotropy choice without an external mass sheet correction.
        This routine computes the IFU map of the kinematic quantities.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with the deflector galaxy, optional,
         if set to None will be computed in this function with default settings that may not be accurate.
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        sigma_v_map = self.velocity_dispersion_map(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light,
                                               kwargs_anisotropy=kwargs_anisotropy, r_eff=r_eff, theta_E=theta_E,
                                               gamma=gamma)
        sigma_v_map *= 1000  # convert from [km/s] to  [m/s]
        J_map = sigma_v_map ** 2 * self._lens_cosmo.dds / self._lens_cosmo.ds / const.c ** 2
        return J_map

    @staticmethod
    def ddt_from_time_delay(d_fermat_model, dt_measured, kappa_s=0, kappa_ds=0, kappa_d=0):
        """
        Time-delay distance in units of Mpc from the modeled Fermat potential and measured time delay from an image pair.

        :param d_fermat_model: relative Fermat potential between two images from the same source in units arcsec^2
        :param dt_measured: measured time delay between the same image pair in units of days
        :return: D_dt, time-delay distance
        """
        D_dt_model = dt_measured * const.day_s * const.c / const.Mpc / d_fermat_model / const.arcsec ** 2
        D_dt = D_dt_model * (1-kappa_ds) / (1 - kappa_s) / (1 - kappa_d)
        return D_dt

    @staticmethod
    def ds_dds_from_kinematics(sigma_v, J, kappa_s=0, kappa_ds=0):
        """
        computes the estimate of the ratio of angular diameter distances Ds/Dds from the kinematic estimate of the lens
        and the measured dispersion.

        :param sigma_v: velocity dispersion [km/s]
        :param J: dimensionless kinematic constraint (see Birrer et al. 2016, 2019)
        :return: Ds/Dds
        """
        ds_dds_model = (sigma_v * 1000) ** 2 / const.c ** 2 / J
        ds_dds = ds_dds_model * (1 - kappa_ds) / (1 - kappa_s)
        return ds_dds

    def ddt_dd_from_time_delay_and_kinematics(self, d_fermat_model, dt_measured, sigma_v_measured, J, kappa_s=0,
                                              kappa_ds=0, kappa_d=0):
        """

        :param d_fermat_model: relative Fermat potential in units arcsec^2
        :param dt_measured: measured relative time delay [days]
        :param sigma_v_measured: 1-sigma Gaussian uncertainty in the measured velocity dispersion
        :param J: modeled dimensionless kinematic estimate
        :param kappa_s: LOS convergence from observer to source
        :param kappa_ds: LOS convergence from deflector to source
        :param kappa_d: LOS convergence from observer to deflector
        :return: D_dt, D_d
        """
        ddt = self.ddt_from_time_delay(d_fermat_model, dt_measured, kappa_s=kappa_s, kappa_ds=kappa_ds, kappa_d=kappa_d)
        ds_dds = self.ds_dds_from_kinematics(sigma_v_measured, J, kappa_s=kappa_s, kappa_ds=kappa_ds)
        dd = ddt / ds_dds / (1 + self._z_lens)
        return ddt, dd
