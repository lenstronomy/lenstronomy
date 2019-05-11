from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import numpy as np


class FluxRatioLikelihood(object):
    """
    likelihood class for magnification of multiply lensed images
    """

    def __init__(self, point_source_class, lens_model_class, param_class, flux_ratios, flux_ratio_errors,
                 source_type='INF', window_size=0.1, grid_number=100, polar_grid=False, aspect_ratio=0.5):
        """

        :param point_source_class: PointSource class instance
        :param lens_model_class: LensModel class instance
        :param param_class: Param() class instance
        :param flux_ratios: ratio of fluxes of the multiple images (relative to the first appearing)
        :param flux_ratio_errors: errors in the flux ratios (relative to the first appearing
        :param source_type: string, type of source, options are 'INF', 'GAUSSIAN', 'TORUS
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically comute the flux
        """
        self._pointSource = point_source_class
        self._lens_model_class = lens_model_class
        self._param = param_class
        self._flux_ratios = np.array(flux_ratios)
        self._flux_ratio_errors = np.array(flux_ratio_errors)
        self._lens_model_extensions = LensModelExtensions(lensModel=lens_model_class)
        self._source_type = source_type
        self._window_size = window_size
        self._gird_number = grid_number
        self._polar_grid = polar_grid
        self._aspect_ratio = aspect_ratio

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :param kwargs_cosmo:
        :return: log likelihood of the measured flux ratios given a model
        """

        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        x_pos, y_pos = self._param.real_image_positions(ra_image_list[0], dec_image_list[0], kwargs_cosmo)
        if self._source_type is 'INF':
            mag = np.abs(self._lens_model_class.magnification(x_pos, y_pos, kwargs_lens))
        else:
            source_sigma = kwargs_cosmo['source_size']
            mag = self._lens_model_extensions.magnification_finite(x_pos, y_pos, kwargs_lens, source_sigma=source_sigma,
                                                                   window_size=self._window_size,
                                                                   grid_number=self._gird_number,
                                                                   shape=self._source_type, polar_grid=self._polar_grid,
                                                                   aspect_ratio=self._aspect_ratio)

        mag_ratio = mag[1:] / mag[0]
        dist = (mag_ratio - self._flux_ratios) ** 2 / self._flux_ratio_errors ** 2 / 2
        logL = -np.sum(dist)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL
