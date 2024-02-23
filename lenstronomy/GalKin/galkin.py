from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.GalKin.galkin_model import GalkinModel

import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import interp1d

__all__ = ["Galkin"]


class Galkin(GalkinModel, GalkinObservation):
    """Major class to compute velocity dispersion measurements given light and mass
    models.

    The class supports any mass and light distribution (and superposition thereof) that has a 3d correspondance in their
    2d lens model distribution. For models that do not have this correspondance, you may want to apply a
    Multi-Gaussian Expansion (MGE) on their models and use the MGE to be de-projected to 3d.

    The computation follows Mamon&Lokas 2005 and performs the spectral rendering of the seeing convolved apperture with
    the method introduced by Birrer et al. 2016.

    The class supports various types of anisotropy models (see Anisotropy class) and aperture types
    (see Aperture class).

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

    def __init__(
        self,
        kwargs_model,
        kwargs_aperture,
        kwargs_psf,
        kwargs_cosmo,
        kwargs_numerics=None,
        analytic_kinematics=False,
    ):
        """

        :param kwargs_model: keyword arguments describing the model components
        :param kwargs_aperture: keyword arguments describing the spectroscopic aperture, see Aperture() class
        :param kwargs_psf: keyword argument specifying the PSF of the observation
        :param kwargs_cosmo: keyword arguments that define the cosmology in terms of the angular diameter distances
         involved
        :param kwargs_numerics: numerics keyword arguments
        :param analytic_kinematics: bool, if True uses the analytic kinematic model
        """
        GalkinModel.__init__(
            self,
            kwargs_model,
            kwargs_cosmo,
            kwargs_numerics=kwargs_numerics,
            analytic_kinematics=analytic_kinematics,
        )
        GalkinObservation.__init__(
            self, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf
        )

    def dispersion(
        self, kwargs_mass, kwargs_light, kwargs_anisotropy, sampling_number=1000
    ):
        """Computes the averaged LOS velocity dispersion in the slit (convolved)

        :param kwargs_mass: mass model parameters (following lenstronomy lens model
            conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light
            model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to
            anisotropy type chosen. We refer to the Anisotropy() class for details on
            the parameters.
        :param sampling_number: int, number of spectral sampling of the light
            distribution
        :return: integrated LOS velocity dispersion in units [km/s]
        """
        sigma2_IR_sum = 0
        IR_sum = 0
        for i in range(0, sampling_number):
            sigma2_IR, IR = self._draw_one_sigma2(
                kwargs_mass, kwargs_light, kwargs_anisotropy
            )
            sigma2_IR_sum += sigma2_IR
            IR_sum += IR
        sigma_s2_average = sigma2_IR_sum / IR_sum
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.0  # in units of km/s

    def dispersion_map(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        num_kin_sampling=1000,
        num_psf_sampling=100,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param num_kin_sampling: int, number of draws from a kinematic prediction of a
            LOS
        :param num_psf_sampling: int, number of displacements/render from a spectra to
            be displaced on the IFU
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
            sigma2_IR, IR = self.numerics.sigma_s2(
                r, R, kwargs_mass, kwargs_light, kwargs_anisotropy
            )
            for k in range(0, num_psf_sampling):
                x_, y_ = self.displace_psf(x, y)
                bool_ap, ifu_index = self.aperture_select(x_, y_)
                if bool_ap is True:
                    sigma2_IR_sum[ifu_index] += sigma2_IR
                    count_draws[ifu_index] += IR

        sigma_s2_average = sigma2_IR_sum / count_draws
        # apply unit conversion from arc seconds and deflections to physical velocity dispersion in (km/s)
        self.numerics.delete_cache()
        return np.sqrt(sigma_s2_average) / 1000.0  # in units of km/s

    @staticmethod
    def _extract_center(kwargs):
        if not isinstance(kwargs, dict):
            if "center_x" in kwargs[0]:
                return kwargs[0]["center_x"], kwargs[0]["center_y"]
            else:
                return 0, 0
        else:
            if "center_x" in kwargs:
                return kwargs["center_x"], kwargs["center_y"]
            else:
                return 0, 0

    def _delta_pix_xy(self):
        """Get the pixel scale of the grid.

        :return: delta_x, delta_y
        """
        x_grid = self._aperture.x_grid
        y_grid = self._aperture.y_grid
        delta_x = x_grid[0, 1] - x_grid[0, 0]
        delta_y = y_grid[1, 0] - y_grid[0, 0]

        return delta_x, delta_y

    def _get_grid(self, kwargs_mass, supersampling_factor=1):
        """Compute the grid to compute the dispersion map on.

        :param kwargs_mass: keyword arguments of the mass model
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :return: x_grid, y_grid, log10_radial_distance_from_center
        """
        mass_center_x, mass_center_y = self._extract_center(kwargs_mass)

        delta_x, delta_y = self._delta_pix_xy()
        assert np.abs(delta_x) == np.abs(delta_y)

        x_grid = self._aperture.x_grid
        y_grid = self._aperture.y_grid

        new_delta_x = delta_x / supersampling_factor
        new_delta_y = delta_y / supersampling_factor
        x_start = x_grid[0, 0] - delta_x / 2.0 * (1 - 1 / supersampling_factor)
        x_end = x_grid[0, -1] + delta_x / 2.0 * (1 - 1 / supersampling_factor)
        y_start = y_grid[0, 0] - delta_y / 2.0 * (1 - 1 / supersampling_factor)
        y_end = y_grid[-1, 0] + delta_y / 2.0 * (1 - 1 / supersampling_factor)

        xs = np.arange(x_start, x_end * (1 + 1e-6), new_delta_x)
        ys = np.arange(y_start, y_end * (1 + 1e-6), new_delta_y)

        x_grid_supersampled, y_grid_supersmapled = np.meshgrid(xs, ys)

        log10_radial_distance_from_center = np.log10(
            np.sqrt(
                (x_grid_supersampled - mass_center_x) ** 2
                + (y_grid_supersmapled - mass_center_y) ** 2
            )
        )

        return (
            x_grid_supersampled,
            y_grid_supersmapled,
            log10_radial_distance_from_center,
        )

    def _get_convolution_kernel(self, fwhm_factor=3, supersampling_factor=1):
        """Normalized convolution kernel.

        :param delta_x: pixel scale of kernel
        :param delta_y: pixel scale of kernel
        :param fwhm_factor: number of FWHM to compute the kernel on
        :param supersampling_factor: number of sub-pixels to compute the kernel on
        """
        delta_x, delta_y = self._delta_pix_xy()
        psf_x = np.arange(
            -fwhm_factor * self._psf.fwhm,
            fwhm_factor * self._psf.fwhm + np.abs(delta_x) / (supersampling_factor + 1),
            np.abs(delta_x) / supersampling_factor,
        )
        psf_y = np.arange(
            -fwhm_factor * self._psf.fwhm,
            fwhm_factor * self._psf.fwhm + np.abs(delta_y) / (supersampling_factor + 1),
            np.abs(delta_y) / supersampling_factor,
        )

        psf_x_grid, psf_y_grid = np.meshgrid(psf_x, psf_y)

        return self.convolution_kernel_grid(psf_x_grid, psf_y_grid)

    def dispersion_map_grid_convolved(
        self,
        kwargs_mass,
        kwargs_light,
        kwargs_anisotropy,
        supersampling_factor=1,
        voronoi_bins=None,
    ):
        """Computes the velocity dispersion in each Integral Field Unit.

        :param kwargs_mass: keyword arguments of the mass model
        :param kwargs_light: keyword argument of the light model
        :param kwargs_anisotropy: anisotropy keyword arguments
        :param supersampling_factor: sampling factor for the grid to do the 2D
            convolution on
        :param voronoi_bins: mapping of the voronoi bins, bin indices should start from
            0, -1 values for pixels not binned
        :return: ordered array of velocity dispersions [km/s] for each unit
        """
        if hasattr(self.numerics, "lum_weight_int_method"):
            if not self.numerics.lum_weight_int_method:
                raise ValueError("'lum_weight_int_method' must be True!")

        (
            x_grid_supersmapled,
            y_grid_supersmapled,
            log10_radial_distance_from_center,
        ) = self._get_grid(kwargs_mass, supersampling_factor=supersampling_factor)

        x_grid = self._aperture.x_grid
        y_grid = self._aperture.y_grid

        mass_center_x, mass_center_y = self._extract_center(kwargs_mass)
        R_max = np.sqrt(
            (x_grid_supersmapled - mass_center_x) ** 2
            + (y_grid_supersmapled - mass_center_y) ** 2
        ).max()
        Rs = np.logspace(
            np.log10(self.numerics.min_integrate), np.log10(R_max + 0.1), 300
        )
        sigma2_IRs = np.zeros_like(Rs)
        IRs = np.zeros_like(Rs)

        for i, R in enumerate(Rs):
            sigma2_IRs[i], IRs[i] = self.numerics.I_R_sigma2_and_IR(
                R, kwargs_mass, kwargs_light, kwargs_anisotropy
            )

        log10_Rs = np.log10(Rs)
        log10_sigma2_IRs = np.log10(sigma2_IRs)
        log10_IRs = np.log10(IRs)

        log10_sigma2_interp = interp1d(
            log10_Rs,
            log10_sigma2_IRs,
            kind="linear",
            bounds_error=False,
            fill_value=(log10_sigma2_IRs[0], log10_sigma2_IRs[-1]),
            assume_sorted=True,
        )
        log10_IR_interp = interp1d(
            log10_Rs,
            log10_IRs,
            kind="linear",
            bounds_error=False,
            fill_value=(log10_IRs[0], log10_IRs[-1]),
            assume_sorted=True,
        )

        sigma2_IR_grid = 10 ** log10_sigma2_interp(log10_radial_distance_from_center)
        IR_grid = 10 ** log10_IR_interp(log10_radial_distance_from_center)

        convolution_kernel = self._get_convolution_kernel(
            fwhm_factor=3, supersampling_factor=supersampling_factor
        )

        sigma2_IR_convolved = convolve2d(
            sigma2_IR_grid, convolution_kernel, mode="same"
        )
        IR_convolved = convolve2d(IR_grid, convolution_kernel, mode="same")

        if voronoi_bins is not None:
            n_bins = int(np.max(voronoi_bins)) + 1

            sigma_IR_integrated = np.zeros(n_bins)
            IR_integrated = np.zeros(n_bins)

            supersampled_voronoi_bins = voronoi_bins.repeat(
                supersampling_factor, axis=0
            ).repeat(supersampling_factor, axis=1)

            for n in range(n_bins):
                sigma_IR_integrated[n] = np.sum(
                    sigma2_IR_convolved[supersampled_voronoi_bins == n]
                )
                IR_integrated[n] = np.sum(IR_convolved[supersampled_voronoi_bins == n])
        else:
            sigma_IR_integrated = (
                sigma2_IR_convolved.reshape(
                    len(x_grid), supersampling_factor, len(y_grid), supersampling_factor
                )
                .sum(3)
                .sum(1)
            )
            IR_integrated = (
                IR_convolved.reshape(
                    len(x_grid), supersampling_factor, len(y_grid), supersampling_factor
                )
                .sum(3)
                .sum(1)
            )

        sigma2_grid = sigma_IR_integrated / IR_integrated

        # apply unit conversion from arc seconds and deflections to physical velocity
        # dispersion in (km/s)
        return np.sqrt(sigma2_grid) / 1000.0  # in units of km/s

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
            bool_ap, _ = self.aperture_select(x_, y_)
            if bool_ap is True:
                break
        sigma2_IR, IR = self.numerics.sigma_s2(
            r, R, kwargs_mass, kwargs_light, kwargs_anisotropy
        )
        return sigma2_IR, IR
