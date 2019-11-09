__author__ = 'sibirrer'


import numpy as np
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import default_cosmology


class TDCosmography(object):
    """
    class equipped to perform a cosmographic analysis from a lens model with added measurements of time delays and
    kinematics.
    This class does not require any cosmological knowledge and can return angular diameter distance estimates
    self-consistently integrating the kinematics routines and time delay estimates in the lens modeling.
    This description follows Birrer et al. 2016, 2019.


    """
    def __init__(self, z_lens, z_source, kwargs_model):

        self._cosmo_fiducial = default_cosmology.get()
        self._lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=self._cosmo_fiducial)
        kwargs_model['cosmo'] = self._cosmo_fiducial  # here we over-write a possible cosmology model to make sure that it is always the same default one to be subtracted off

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
