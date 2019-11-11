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
    def __init__(self, z_lens, z_source, kwargs_model, cosmo_fiducial=None, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None):

        if cosmo_fiducial is None:
            cosmo_fiducial = default_cosmology.get()
        self._cosmo_fiducial = cosmo_fiducial
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo_fiducial)
        self._kinematic_api = KinematicAPI(z_lens, z_source, kwargs_model, cosmo=self._cosmo_fiducial,
                                           lens_model_kinematics_bool=lens_model_kinematics_bool,
                                           light_model_kinematics_bool=light_model_kinematics_bool)
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
        fermat_pot = self.LensModel.fermat_potential(ra_pos, dec_pos, kwargs_lens, ra_source, dec_source)
        return fermat_pot

    def velocity_dispersion_dimension_less(self, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=None,
                                           theta_E=None, gamma=None):
        """
        \sigma^2 = D_d/D_ds * c^2 *J(kwargs_lens, kwargs_light, anisotropy) (Equation 4.11 in Birrer et al. 2016 or Equation 6 in Birrer et al. 2019)
        J() is a dimensionless and cosmological independent quantity only depending on angular units
        This function returns J given the lens and light parameters and the anisotropy choice without an external mass sheet correction.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_lens_light: lens light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :param r_eff: projected half-light radius of the stellar light associated with the deflector galaxy, optional,
         if set to None will be computed in this function with default settings that may not be accurate.
        :return: dimensionless velocity dispersion (see e.g. Birrer et al. 2016, 2019)
        """
        sigma_v = self._kinematic_api.model_velocity_dispersion(kwargs_lens=kwargs_lens,
                                                                kwargs_lens_light=kwargs_lens_light,
                                                                kwargs_anisotropy=kwargs_anisotropy,
                                                                r_eff=r_eff, theta_E=theta_E, gamma=gamma)
        J = sigma_v ** 2 * self._lens_cosmo.D_ds / self._lens_cosmo.D_s / const.c ** 2
        return J

    @staticmethod
    def D_dt_from_time_delay(d_fermat_model, dt_measured, kappa_s=0, kappa_ds=0, kappa_d=0):
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
    def Ds_Dds_from_kinematics(sigma_v, J, kappa_s=0, kappa_ds=0):
        """
        computes the estimate of the ratio of angular diameter distances Ds/Dds from the kinematic estimate of the lens
        and the measured dispersion.

        :param sigma_v: velocity dispersion [km/s]
        :param J: dimensionless kinematic constraint (see Birrer et al. 2016, 2019)
        :return: Ds/Dds
        """
        Ds_Dds_model = (sigma_v / 1000) ** 2 / const.c ** 2 / J
        Ds_Dds = Ds_Dds_model * (1 - kappa_ds) / (1 - kappa_s)
        return Ds_Dds
