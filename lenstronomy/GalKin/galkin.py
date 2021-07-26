from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.galkin_model import GalkinModel

import numpy as np

__all__ = ['Galkin']


class Galkin(GalkinModel, GalkinObservation):
    """
    Major class to compute velocity dispersion measurements given light and mass models

    The class supports any mass and light distribution (and superposition thereof) that has a 3d correspondance in their
    2d lens model distribution. For models that do not have this correspondance, you may want to apply a
    Multi-Gaussian Expansion (MGE) on their models and use the MGE to be de-projected to 3d.

    The computation follows Mamon&Lokas 2005 and performs the spectral rendering of the seeing convolved apperture with
    the method introduced by Birrer et al. 2016.

    The class supports various types of anisotropy models (see Anisotropy class) and aperture types (see Aperture class).
    Solving the Jeans Equation requires a numerical integral over the 3d light and mass profile (see Mamon&Lokas 2005).
    This class (as well as the dedicated LightModel and MassModel classes) perform those integral numerically with an
    interpolated grid.

    The seeing convolved integral over the aperture is computed by rendering spectra (light weighted LOS kinematics)
    from the light distribution.

    The cosmology assumed to compute the physical mass and distances are set via the kwargs_cosmo keyword arguments.
        d_d: Angular diameter distance to the deflector (in Mpc)
        d_s: Angular diameter distance to the source (in Mpc)
        d_ds: Angular diameter distance from the deflector to the source (in Mpc)

    The numerical options can be chosen through the kwargs_numerics keywords

        interpol_grid_num: number of interpolation points in the light and mass profile (radially). This number should
        be chosen high enough to accurately describe the true light profile underneath.
        log_integration: bool, if True, performs the interpolation and numerical integration in log-scale.

        max_integrate: maximum 3d radius to where the numerical integration of the Jeans Equation solver is made.
        This value should be large enough to contain most of the light and to lead to a converged result.
        min_integrate: minimal integration value. This value should be very close to zero but some mass and light
        profiles are diverging and a numerically stabel value should be chosen.

    These numerical options should be chosen to allow for a converged result (within your tolerance) but not too
    conservative to impact too much the computational cost. Reasonable values might depend on the specific problem.

    """
    def __init__(self, kwargs_model, kwargs_aperture, kwargs_psf, kwargs_cosmo, kwargs_numerics=None,
                 analytic_kinematics=False):
        """

        :param kwargs_model: keyword arguments describing the model components
        :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param kwargs_psf: keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances involved
        :param kwargs_numerics: numerics keyword arguments
        :param analytic_kinematics: bool, if True uses the analytic kinematic model
        """
        GalkinModel.__init__(self, kwargs_model, kwargs_cosmo, kwargs_numerics=kwargs_numerics,
                             analytic_kinematics=analytic_kinematics)
        GalkinObservation.__init__(self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf)

    def dispersion(self, kwargs_mass, kwargs_light, kwargs_anisotropy, sampling_number=1000):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :param sampling_number: int, number of spectral sampling of the light distribution
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        sigma2_IR_sum = 0
        IR_sum = 0
        for i in range(0, sampling_number):
            sigma2_IR, IR = self._draw_one_sigma2(kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma2_IR_sum += sigma2_IR
            IR_sum += IR
        sigma_s2_average = sigma2_IR_sum / IR_sum
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.  # in units of km/s

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
        sigma2_IR_sum = np.zeros(num_segments)
        count_draws = np.zeros(num_segments)

        for i in range(0, num_kin_sampling):
            r, R, x, y = self.numerics.draw_light(kwargs_light)
            sigma2_IR, IR = self.numerics.sigma_s2(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            for k in range(0, num_psf_sampling):
                x_, y_ = self.displace_psf(x, y)
                bool, ifu_index = self.aperture_select(x_, y_)
                if bool is True:
                    sigma2_IR_sum[ifu_index] += sigma2_IR
                    count_draws[ifu_index] += IR

        sigma_s2_average = sigma2_IR_sum / count_draws
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.  # in units of km/s

    def _draw_one_sigma2(self, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in angular units for a single draw of the light distribution that
         falls in the aperture after displacing with the seeing
        """
        while True:
            r, R, x, y = self.numerics.draw_light(kwargs_light)
            x_, y_ = self.displace_psf(x, y)
            bool, _ = self.aperture_select(x_, y_)
            if bool is True:
                break
        sigma2_IR, IR = self.numerics.sigma_s2(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        return sigma2_IR, IR
