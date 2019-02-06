__author__ = 'sibirrer'

from lenstronomy.ImSim.multi_source_plane import MultiSourcePlane
from lenstronomy.ImSim.image_model import ImageModel
import numpy as np


class ImageModelMultiSource(ImageModel):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, kwargs_numerics={}):
        """
        :param data_class: instance of Data() class
        :param psf_class: instance of PSF() class
        :param lens_model_class: instance of LensModel() class
        :param source_model_class: instance of LightModel() class describing the source parameters
        :param lens_light_model_class: instance of LightModel() class describing the lens light parameters
        :param point_source_class: instance of PointSource() class describing the point sources
        :param kwargs_numerics: keyword argument with various numeric description (see ImageNumerics class for options)
        """
        self.source_mapping = MultiSourcePlane(lensModel=lens_model_class,
                                                   light_model_list=source_model_class.profile_type_list,
                                                   source_scale_factor_list=source_model_class.deflection_scaling_list,
                                               source_redshift_list=source_model_class.redshift_list)
        super(ImageModelMultiSource, self).__init__(data_class, psf_class, lens_model_class, source_model_class,
                                         lens_light_model_class, point_source_class, kwargs_numerics)

    def source_surface_brightness(self, kwargs_source, kwargs_lens=None, unconvolved=False, de_lensed=False, k=None):
        """

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: 1d array of surface brightness pixels
        """
        if self.SourceModel is None:
            return np.zeros_like(self.Data.data)
        if de_lensed is True:
            x_source, y_source = self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting
            source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source, k=k)
        else:
            source_light = self.source_mapping.ray_trace_joint(self.ImageNumerics.ra_grid_ray_shooting,
                                                               self.ImageNumerics.dec_grid_ray_shooting, kwargs_lens,
                                                               kwargs_source)
        source_light_final = self.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return source_light_final

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        A = self._response_matrix_new(self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting,
                                      kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, self.ImageNumerics.mask)
        return A

    def _response_matrix_new(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, mask, unconvolved=False):
        """

        return linear response Matrix

        :param x_grid:
        :param y_grid:
        :param x_source:
        :param y_source:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param mask:
        :param unconvolved:
        :return:
        """
        if not self.SourceModel is None:
            source_light_response, n_source = self.source_mapping.ray_trace_functions_split(x_grid, y_grid, kwargs_lens,
                                                                                            kwargs_source)
        else:
            source_light_response, n_source = [], 0
        if not self.LensLightModel is None:
            lens_light_response, n_lens_light = self.LensLightModel.functions_split(x_grid, y_grid, kwargs_lens_light)
        else:
            lens_light_response, n_lens_light = [], 0
        if not self.PointSource is None:
            ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=False)
        else:
            ra_pos, dec_pos, amp, n_points = [], [], [], 0
        num_param = n_points + n_lens_light + n_source

        num_response = self.ImageNumerics.num_response
        A = np.zeros((num_param, num_response))
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        # response of point sources
        for i in range(0, n_points):
            image = self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        A = self._add_mask(A, mask)
        return A
