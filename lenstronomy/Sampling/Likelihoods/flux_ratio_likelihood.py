from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import numpy as np

__all__ = ["FluxRatioLikelihood"]


class FluxRatioLikelihood(object):
    """Likelihood class for magnification of multiply lensed images."""

    def __init__(
        self,
        lens_model_class,
        flux_ratios,
        flux_ratio_errors,
        flux_ratio_measurement_bool=None,
        num_point_sources=1,
        source_type="INF",
        window_size=0.1,
        grid_number=100,
        polar_grid=False,
        aspect_ratio=0.5,
        point_source_redshift_list=None,
    ):
        """

        :param lens_model_class: LensModel class instance
        :param flux_ratios: ratio of fluxes of the multiple images (relative to the first appearing in
         same order as the images)
        :param flux_ratio_errors: errors in the flux ratios (relative to the first appearing). Alternatively
         a log-normal covariance matrix. Note: in the case of a covariance matrix, the errors are assumed
         to be log-normal, i.e. the logarithms of the flux ratios, ln(F[i]/F[0]) are assumed to have a multivariate
         Gaussian distribution, with the given covariance matrix.
        :param flux_ratio_measurement_bool: list of bools of length of the point source model,
         indicating for which point sources there is a flux ratio measurement
        :param num_point_sources: number of point sources in the point source model output
        :type num_point_sources: int>=0
        :param source_type: string, type of source, 'INF' specifies a point source, while 'GAUSSIAN' specifies a
         finite-size source modeled as a Gaussian
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically compute the flux
        :param point_source_redshift_list: list of redshifts to the different sources
        """
        self._num_point_sources = num_point_sources
        self._lens_model_class = lens_model_class
        if num_point_sources == 1:
            self._flux_ratios = [np.array(flux_ratios)]
            self._flux_ratio_errors = [np.array(flux_ratio_errors)]
        else:
            self._flux_ratios = []
            self._flux_ratio_errors = []
            for i in range(num_point_sources):
                self._flux_ratios.append(np.array(flux_ratios[i]))
                self._flux_ratio_errors.append(np.array(flux_ratio_errors[i]))
        self._lens_model_extensions = LensModelExtensions(lensModel=lens_model_class)
        self._source_type = source_type
        self._window_size = window_size
        self._gird_number = grid_number
        self._polar_grid = polar_grid
        self._aspect_ratio = aspect_ratio
        self._num_point_sources = num_point_sources
        if flux_ratio_measurement_bool is None:
            flux_ratio_measurement_bool = [True] * num_point_sources
        self._flux_ratio_measurement_bool = flux_ratio_measurement_bool
        if point_source_redshift_list is None:
            point_source_redshift_list = [None] * num_point_sources
        self._point_source_redshift_list = point_source_redshift_list

    def logL(self, ra_image_list, dec_image_list, kwargs_lens, kwargs_special):
        """

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_special: dictionary of 'special' keyword parameters
        :return: log likelihood of the measured flux ratios given a model
        """
        log_l = 0
        for k in range(len(ra_image_list)):
            if self._flux_ratio_measurement_bool[k] is True:
                x_pos, y_pos = ra_image_list[k], dec_image_list[k]
                self._lens_model_class.change_source_redshift(
                    z_source=self._point_source_redshift_list[k]
                )
                if self._source_type == "INF":
                    mag = np.abs(
                        self._lens_model_class.magnification(x_pos, y_pos, kwargs_lens)
                    )
                else:
                    source_sigma = kwargs_special["source_size"]
                    mag = self._lens_model_extensions.magnification_finite(
                        x_pos,
                        y_pos,
                        kwargs_lens,
                        source_sigma=source_sigma,
                        window_size=self._window_size,
                        grid_number=self._gird_number,
                        polar_grid=self._polar_grid,
                        aspect_ratio=self._aspect_ratio,
                    )
                if len(mag) - 1 != len(self._flux_ratios[k]):
                    return -(10**15)
                mag_ratio = mag[1:] / mag[0]
                log_l += self._logL(mag_ratio, k=k)
        return log_l

    def _logL(self, flux_ratios, k=0):
        """


        :param flux_ratios: flux ratios from the model
        :param k: integer listing the n'th point source model
        :return: log likelihood fo the measured flux ratios given a model
        """
        flux_ratios_obs = self._flux_ratios[k]
        flux_ratio_errors = self._flux_ratio_errors[k]
        if not np.isfinite(flux_ratios).any():
            return -(10**15)
        if flux_ratio_errors.ndim <= 1:
            dist = (flux_ratios - flux_ratios_obs) ** 2 / flux_ratio_errors**2 / 2
            logL = -np.sum(dist)
        elif flux_ratio_errors.ndim == 2:
            # Assume covariance matrix is in ln units!
            D = np.log(flux_ratios) - np.log(flux_ratios_obs)
            logL = (
                -1 / 2 * D @ np.linalg.inv(flux_ratio_errors) @ D
            )  # TODO: only calculate the inverse once
        else:
            raise ValueError(
                "flux_ratio_errors need dimension of 1 or 2. Current dimensions are %s"
                % self._flux_ratio_errors.ndim
            )
        if not np.isfinite(logL):
            return -(10**15)
        return logL

    @property
    def num_data(self):
        """

        :return: integer, number of data points associated with the flux ratios
        """
        num = 0
        for k in range(self._num_point_sources):
            if self._flux_ratio_measurement_bool[k] is True:
                num += len(self._flux_ratios[k])
        return num
