__author__ = 'sibirrer'

from lenstronomy.ImSim.image_numerics import ImageNumerics
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

import numpy as np


class ImageModel(object):
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
        self.type = 'single-band'
        self.PSF = psf_class
        self.Data = data_class
        self.ImageNumerics = ImageNumerics(pixel_grid=self.Data, psf=self.PSF, **kwargs_numerics)
        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class
        self.PointSource = point_source_class
        self._error_map_bool_list = None
        if self.PointSource is not None:
            self.PointSource.update_lens_model(lens_model_class=lens_model_class)
            x_center, y_center = self.Data.center
            self.PointSource.update_search_window(search_window=np.max(self.Data.width), x_center=x_center,
                                                  y_center=y_center, min_distance=self.Data.pixel_width)
            if self.PSF.psf_error_map is not None:
                self._psf_error_map = True
                self._error_map_bool_list = kwargs_numerics.get('error_map_bool_list', [True]*len(self.PointSource.point_source_type_list))
            else:
                self._psf_error_map = False
        else:
            self._psf_error_map = False
        if source_model_class is None:
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        self.LensLightModel = lens_light_model_class
        self.source_mapping = Image2SourceMapping(lensModel=lens_model_class, sourceModel=source_model_class)
        self.num_bands = 1

    def reset_point_source_cache(self, bool=True):
        """
        deletes all the cache in the point source class and saves it from then on

        :return:
        """
        if self.PointSource is not None:
            self.PointSource.delete_lens_model_cach()
            self.PointSource.set_save_cache(bool)

    def update_data(self, data_class):
        """

        :param data_class: instance of Data() class
        :return: no return. Class is updated.
        """
        self.Data = data_class
        self.ImageNumerics._PixelGrid = data_class

    def update_psf(self, psf_class):
        """

        update the instance of the class with a new instance of PSF() with a potentially different point spread function

        :param psf_class:
        :return: no return. Class is updated.
        """
        self.PSF = psf_class
        self.ImageNumerics._PSF = psf_class

    def source_surface_brightness(self, kwargs_source, kwargs_lens=None, unconvolved=False, de_lensed=False, k=None):
        """

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: 1d array of surface brightness pixels
        """
        if len(self.SourceModel.profile_type_list) == 0:
            return np.zeros_like(self.Data.data)
        if de_lensed is True:
            x_source, y_source = self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting
            source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source, k=k)
        else:
            source_light = self.source_mapping.image_flux_joint(self.ImageNumerics.ra_grid_ray_shooting,
                                                                self.ImageNumerics.dec_grid_ray_shooting, kwargs_lens,
                                                                kwargs_source, k=k)
        source_light_final = self.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """

        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: 1d array of surface brightness pixels
        """
        if self.LensLightModel is None:
            return np.zeros_like(self.Data.data)
        lens_light = self.LensLightModel.surface_brightness(self.ImageNumerics.ra_grid_ray_shooting,
                                                            self.ImageNumerics.dec_grid_ray_shooting,
                                                            kwargs_lens_light, k=k)
        lens_light_final = self.ImageNumerics.re_size_convolve(lens_light, unconvolved=unconvolved)
        return lens_light_final

    def point_source(self, kwargs_ps, kwargs_lens=None, unconvolved=False, k=None):
        """

        computes the point source positions and paints PSF convolutions on them

        :param kwargs_ps:
        :param k:
        :return:
        """
        point_source_image = np.zeros_like(self.Data.data)
        if unconvolved or self.PointSource is None:
            return point_source_image
        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=True, k=k)
        for i in range(n_points):
            point_source_image += self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
        return point_source_image

    def image(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, unconvolved=False,
              source_add=True, lens_light_add=True, point_source_add=True):
        """

        make a image with a realisation of linear parameter values "param"

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 1d array of surface brightness pixels of the simulation
        """
        if source_add:
            source_light = self.source_surface_brightness(kwargs_source, kwargs_lens, unconvolved=unconvolved)
        else:
            source_light = np.zeros_like(self.Data.data)
        if lens_light_add:
            lens_light = self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        else:
            lens_light = np.zeros_like(self.Data.data)
        if point_source_add:
            point_source = self.point_source(kwargs_ps, kwargs_lens, unconvolved=unconvolved)
        else:
            point_source = np.zeros_like(self.Data.data)
        model = (source_light + lens_light + point_source) * self.ImageNumerics.mask
        return model

    def error_map(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        error_map = np.zeros_like(self.Data.data)
        if self._psf_error_map is True:
            for k, bool in enumerate(self._error_map_bool_list):
                if bool is True:
                    ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, k=k)
                    for i in range(0, n_points):
                        error_map_add = self.ImageNumerics.psf_error_map(ra_pos[i], dec_pos[i], amp[i], self.Data.data)
                        error_map += error_map_add
        return error_map

    def reduced_residuals(self, model, error_map=0):
        """

        :param model:
        :return:
        """
        mask = self.ImageNumerics.mask
        residual = (model - self.Data.data)/np.sqrt(self.Data.C_D+np.abs(error_map))*mask
        return residual

    def reduced_chi2(self, model, error_map=0):
        """
        returns reduced chi2
        :param model:
        :param error_map:
        :return:
        """
        chi2 = self.reduced_residuals(model, error_map)
        return np.sum(chi2**2) / self.num_data_evaluate()

    def num_data_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return self.ImageNumerics.numData_evaluate
