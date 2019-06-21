__author__ = 'sibirrer'

from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource

import numpy as np


class ImageModel(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, kwargs_numerics={}):
        """
        :param data_class: instance of ImageData() or PixelGrid() class
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
        self.PSF.set_pixel_size(self.Data.pixel_width)
        self.ImageNumerics = NumericsSubFrame(pixel_grid=self.Data, psf=self.PSF, **kwargs_numerics)
        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class
        if point_source_class is None:
            point_source_class = PointSource(point_source_type_list=[])
        self.PointSource = point_source_class
        if self.PointSource is not None:
            self.PointSource.update_lens_model(lens_model_class=lens_model_class)
            x_center, y_center = self.Data.center
            self.PointSource.update_search_window(search_window=np.max(self.Data.width), x_center=x_center,
                                                  y_center=y_center, min_distance=self.Data.pixel_width)
            if self.PSF.psf_error_map is not None:
                self._psf_error_map = True
            else:
                self._psf_error_map = False
        else:
            self._psf_error_map = False
        if source_model_class is None:
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        if lens_light_model_class is None:
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class
        self.source_mapping = Image2SourceMapping(lensModel=lens_model_class, sourceModel=source_model_class)
        self.num_bands = 1
        self._kwargs_numerics = kwargs_numerics

    def reset_point_source_cache(self, bool=True):
        """
        deletes all the cache in the point source class and saves it from then on

        :param bool: boolean, if True, saves the next occuring point source positions in the cache
        :return: None
        """
        self.PointSource.delete_lens_model_cache()
        self.PointSource.set_save_cache(bool)

    def update_psf(self, psf_class):
        """

        update the instance of the class with a new instance of PSF() with a potentially different point spread function

        :param psf_class:
        :return: no return. Class is updated.
        """
        self.PSF = psf_class
        self.PSF.set_pixel_size(self.Data.pixel_width)
        self.ImageNumerics = NumericsSubFrame(pixel_grid=self.Data, psf=self.PSF, **self._kwargs_numerics)

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
            return np.zeros((self.Data.num_pixel_axes))
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(ra_grid, dec_grid, kwargs_source, k=k)
        else:
            source_light = self.source_mapping.image_flux_joint(ra_grid, dec_grid, kwargs_lens, kwargs_source, k=k)
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
            return np.zeros((self.Data.num_pixel_axes))
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(ra_grid, dec_grid, kwargs_lens_light, k=k)
        lens_light_final = self.ImageNumerics.re_size_convolve(lens_light, unconvolved=unconvolved)
        return lens_light_final

    def point_source(self, kwargs_ps, kwargs_lens=None, unconvolved=False, k=None):
        """

        computes the point source positions and paints PSF convolutions on them

        :param kwargs_ps:
        :param k:
        :return:
        """
        point_source_image = np.zeros((self.Data.num_pixel_axes))
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
        model = np.zeros((self.Data.num_pixel_axes))
        if source_add is True:
            model += self.source_surface_brightness(kwargs_source, kwargs_lens, unconvolved=unconvolved)
        if lens_light_add is True:
            model += self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        if point_source_add is True:
            model += self.point_source(kwargs_ps, kwargs_lens, unconvolved=unconvolved)
        return model
