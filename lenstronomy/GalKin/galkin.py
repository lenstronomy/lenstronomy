from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.numeric_kinematics import NumericKinematics
import lenstronomy.GalKin.velocity_util as util
import lenstronomy.Util.constants as const

import numpy as np


class Galkin(GalkinObservation, NumericKinematics):
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
        D_d: Angular diameter distance to the deflector (in Mpc)
        D_s: Angular diameter distance to the source (in Mpc)
        D_ds: Angular diameter distance from the deflector to the source (in Mpc)

    The numerical options can be chosen through the kwargs_numerics keywords
        sampling_number: number of spectral rendering to compute the light weighted integrated LOS dispersion within
        the aperture. This keyword should be chosen high enough to result in converged results within the tolerance.

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
    def __init__(self, mass_profile_list, light_profile_list, kwargs_aperture, kwargs_psf, anisotropy_model='isotropic',
                 kwargs_cosmo={'d_d': 1000, 'd_s': 2000, 'd_ds': 500},
                 sampling_number=1000, interpol_grid_num=500, log_integration=False, max_integrate=10, min_integrate=0.001):
        """

        :param mass_profile_list: list of lens (mass) model profiles
        :param light_profile_list: list of light model profiles of the lensing galaxy
        :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param anisotropy_model: type of stellar anisotropy model. See details in MamonLokasAnisotropy() class.
        :param kwargs_psf: keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances involved
        """
        NumericKinematics.__init__(self, mass_profile_list, light_profile_list, anisotropy_model=anisotropy_model,
                 kwargs_cosmo=kwargs_cosmo, interpol_grid_num=interpol_grid_num, log_integration=log_integration,
                 max_integrate=max_integrate, min_integrate=min_integrate)
        GalkinObservation.__init__(self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf)

        self.cosmo = Cosmo(**kwargs_cosmo)
        self._num_sampling = sampling_number

    def vel_disp(self, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        sigma2_R_sum = 0
        for i in range(0, self._num_sampling):
            sigma2_R = self._draw_one_sigma2(kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma2_R_sum += sigma2_R
        sigma_s2_average = sigma2_R_sum / self._num_sampling
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        sigma_s2_average *= 2 * const.G  # correcting for integral prefactor
        return np.sqrt(sigma_s2_average / (const.arcsec ** 2 * self.cosmo.dd ** 2 * const.Mpc)) / 1000.  # in units of km/s

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
            R = self.lightProfile.draw_light_2d(kwargs_light, n=1)[0]  # draw r in arcsec
            x, y = util.draw_xy(R)  # draw projected R in arcsec
            x_, y_ = self.displace_psf(x, y)
            bool = self.aperture_select(x_, y_)
            if bool is True:
                break
        sigma2_R = self._sigma2_R(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        return sigma2_R
