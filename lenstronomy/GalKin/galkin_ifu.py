import numpy as np

from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.numeric_kinematics import NumericKinematics
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics


class GalkinIFU(GalkinObservation):
    """
    class to compute the kinematics of an integral field unit
    """
    def __init__(self, kwargs_aperture, kwargs_psf, kwargs_cosmo, kwargs_model, kwargs_numerics={},
                 analytic_kinematics=False):
        """

        :param kwargs_aperture: keyword arguments of the aperture
        :param kwargs_psf: keyword arguments of the seeing condition
        :param kwargs_cosmo: cosmological distances used in the calculation
        :param kwargs_model: keyword arguments of the models
        :param kwargs_numerics: keyword arguments of the numerical description
        """
        if analytic_kinematics is True:
            self.numerics = AnalyticKinematics(kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                                               kwargs_cosmo=kwargs_cosmo)
        else:
            self.numerics = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        GalkinObservation.__init__(self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf)
        self._analytic_kinematics = analytic_kinematics

    def dispersion_map(self, kwargs_mass, kwargs_light, kwargs_anisotropy, num_kin_sampling=1000, num_psf_sampling=100):
        """
        computes the velocity dispersion in each Integral Field Unit

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param num_kin_sampling: int, number of draws from a kinematic prediction of a LOS
        :param num_psf_sampling: int, number of displacements/render from a spectra to be displaced on the IFU
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        # draw from light profile (3d and 2d option)
        # compute kinematics of it (analytic or numerical)
        # displace it n-times
        # add it and keep track of how many draws are added on each segment
        # compute average in each segment
        # return value per segment
        num_segments = self.num_segments
        sigma2_R_sum = np.zeros(num_segments)
        count_draws = np.zeros(num_segments)

        for i in range(0, num_kin_sampling):
            r, R, x, y = self.numerics.draw_light(kwargs_light)
            sigma2_R = self.numerics.sigma_s2(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            for k in range(0, num_psf_sampling):
                x_, y_ = self.displace_psf(x, y)
                bool, ifu_index = self.aperture_select(x_, y_)
                if bool is True:
                    sigma2_R_sum[ifu_index] += sigma2_R
                    count_draws[ifu_index] += 1

        sigma_s2_average = sigma2_R_sum / count_draws
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.  # in units of km/s        while True:
