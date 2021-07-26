from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.galkin_model import GalkinModel

import numpy as np

__all__ = ['GalkinMultiObservation']


class GalkinMultiObservation(GalkinModel):
    """
    class to efficiently model the velocity dispersion measurement of a set of different observations
    with individual apertures and seeing conditions for a given lens

    The main difference to the Galkin main class is that it feeds in list of observational settings.
    Does not work with IFU observations (yet)
    """
    def __init__(self, kwargs_model, kwargs_aperture_list, kwargs_psf_list, kwargs_cosmo, kwargs_numerics=None,
                 analytic_kinematics=False):
        """

        :param kwargs_model: keyword arguments describing the model components
        :param kwargs_aperture_list: list of keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param kwargs_psf_list: list of keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances involved
        :param kwargs_numerics: numerics keyword arguments - see GalkinModel
        :param analytic_kinematics: bool, if True uses the analytic kinematic model
        """
        GalkinModel.__init__(self, kwargs_model, kwargs_cosmo, kwargs_numerics=kwargs_numerics,
                             analytic_kinematics=analytic_kinematics)
        self._observation_list = []
        self._num_observations = len(kwargs_aperture_list)
        for i in range(self._num_observations):
            self._observation_list.append(GalkinObservation(kwargs_aperture=kwargs_aperture_list[i], kwargs_psf=kwargs_psf_list[i]))

    def dispersion_map(self, kwargs_mass, kwargs_light, kwargs_anisotropy, num_kin_sampling=1000, num_psf_sampling=100):
        """
        computes the velocity dispersion in each Integral Field Unit

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param num_kin_sampling: int, number of draws from a kinematic prediction of a LOS
        :param num_psf_sampling: int, number of displacements/render from a spectra to be displaced on the IFU
        :return: ordered array of velocity dispersions [km/s] for each observation
        """
        # draw from light profile (3d and 2d option)
        # compute kinematics of it (analytic or numerical)
        # displace it n-times
        # add it and keep track of how many draws are added on each segment
        # compute average in each segment
        # return value per segment
        sigma2_R_sum = np.zeros(self._num_observations)
        count_draws = np.zeros(self._num_observations)

        for i in range(0, num_kin_sampling):
            r, R, x, y = self.numerics.draw_light(kwargs_light)
            sigma2_IR, IR = self.numerics.sigma_s2(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            for obs_index, observation in enumerate(self._observation_list):
                for k in range(0, num_psf_sampling):
                    x_, y_ = observation.displace_psf(x, y)
                    bool, _ = observation.aperture_select(x_, y_)
                    if bool is True:
                        sigma2_R_sum[obs_index] += sigma2_IR
                        count_draws[obs_index] += IR

        sigma_s2_average = sigma2_R_sum / count_draws
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.  # in units of km/s
