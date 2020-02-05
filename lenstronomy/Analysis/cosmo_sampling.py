import numpy as np
from lenstronomy.Analysis.td_cosmography import TDCosmography


class DsDdsConstraints(object):
    """
    class for sampling Ds/Dds posteriors from imaging data and kinematic constraints
    """

    def __init__(self, z_lens, z_source, theta_E, theta_E_error, gamma, gamma_error, r_eff, r_eff_error, sigma_v,
                 sigma_v_error, kwargs_aperture, kwargs_seeing, kwargs_numerics_galkin, anisotropy_model,
                 kwargs_lens_light=None, lens_light_model_list=['HERNQUIST'], MGE_light=False, kwargs_mge_light=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param theta_E: Einstein radius (in arc seconds)
        :param theta_E_error: 1-sigma error on Einstein radius
        :param gamma: power-law slope (2 = isothermal) estimated from imaging data
        :param gamma_error: 1-sigma uncertainty on power-law slope
        :param r_eff: half-light radius of the deflector (arc seconds)
        :param r_eff_error: uncertainty on half-light radius
        :param sigma_v: velocity dispersion of the main deflecto in km/s
        :param sigma_v_error: 1-sigma uncertainty in velocity dispersion
        :param kwargs_aperture: spectroscopic aperture keyword arguments, see lenstronomy.Galkin.aperture for options
        :param kwargs_seeing: seeing condition of spectroscopic observation, corresponds to kwargs_psf in the GalKin module specified in lenstronomy.GalKin.psf
        :param kwargs_numerics_galkin: numerical settings for the integrated line-of-sight velocity dispersion
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class of lenstronomy.GalKin.anisotropy
        :param kwargs_lens_light: keyword argument list of lens light model (optional)
        :param kwargs_mge_light: keyword arguments that go into the MGE decomposition routine
        """
        kwargs_model = {'lens_model_list': ['SPP'], 'lens_light_model_list': lens_light_model_list}
        self._sigma_v, self._sigma_v_error = sigma_v, sigma_v_error
        self._theta_E, self._theta_E_error = theta_E, theta_E_error
        self._r_eff, self._r_eff_error = r_eff, r_eff_error
        self._gamma, self._gamma_error = gamma, gamma_error
        self._td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=None,
                                 lens_model_kinematics_bool=None, light_model_kinematics_bool=None)
        self._td_cosmo.kinematic_observation_settings(kwargs_aperture, kwargs_seeing)
        if kwargs_lens_light is None:
            analytic_kinematics = True
            hernquist_approx = True
        else:
            analytic_kinematics = False
            hernquist_approx = False
        self._td_cosmo.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin,
                                                    analytic_kinematics=analytic_kinematics,
                                                    Hernquist_approx=hernquist_approx, MGE_light=MGE_light,
                                                    MGE_mass=False, kwargs_mge_light=kwargs_mge_light)
        self._kwargs_lens_light = kwargs_lens_light

    def draw_vel_disp(self, num=1):
        """
        produces realizations of measurements based on the uncertainty in the measurement of the velocity dispersion

        :param num: int, number of realization
        :return: realizations of draws from the measured velocity dispersion
        """
        return np.random.normal(loc=self._sigma_v, scale=self._sigma_v_error, size=num)

    @property
    def draw_lens(self):
        """

        :return: theta_E, gamma, r_eff
        """
        theta_E_draw = np.random.normal(loc=self._theta_E, scale=self._theta_E_error)
        gamma_draw = np.random.normal(loc=self._gamma, scale=self._gamma_error)
        r_eff_draw = np.random.normal(loc=self._r_eff, scale=self._r_eff_error)
        return theta_E_draw, gamma_draw, r_eff_draw

    def ds_dds_realization(self, kwargs_anisotropy, no_error=False):
        """
        creates a realization of Ds/Dds from the measurement uncertainties

        :param kwargs_anisotropy: keyword argument of anisotropy setting
        :param no_error: bool, if True, does not render from the uncertainty but uses the mean values instead
        """

        # compute dimensionless kinematic quantity
        if no_error is True:
            theta_E_draw, gamma_draw, r_eff_draw, sigma_v_draw = self._theta_E, self._gamma, self._r_eff, self._sigma_v
        else:
            theta_E_draw, gamma_draw, r_eff_draw = self.draw_lens
            sigma_v_draw = self.draw_vel_disp(num=1)
        kwargs_lens = [{'theta_E': theta_E_draw, 'gamma': gamma_draw, 'center_x': 0, 'center_y': 0}]
        J = self._td_cosmo.velocity_dispersion_dimension_less(kwargs_lens=kwargs_lens, kwargs_lens_light=self._kwargs_lens_light,
                                                        kwargs_anisotropy=kwargs_anisotropy, r_eff=r_eff_draw,
                                                        theta_E=theta_E_draw, gamma=gamma_draw)
        ds_dds = self._td_cosmo.Ds_Dds_from_kinematics(sigma_v_draw, J, kappa_s=0, kappa_ds=0)
        return ds_dds

    def ds_dds_sample(self, num_sample, kwargs_anisotropy):
        """

        :param num_sample: int, number of samples drawn from the measurement uncertainties
        :param kwargs_anisotropy:
        :return: numpy array of posterior values of Ds/Dds
        """
        ds_dds_list = []
        for i in range(num_sample):
            ds_dds = self.ds_dds_realization(kwargs_anisotropy)
            ds_dds_list.append(ds_dds)
        return np.array(ds_dds_list)
