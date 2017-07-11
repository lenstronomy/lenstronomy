__author__ = 'sibirrer'

from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.ImSim.light_model import LensLightModel, SourceModel
from lenstronomy.ImSim.point_source import PointSource

from lenstronomy.ImSim.data import Data
import lenstronomy.DeLens.de_lens as de_lens

import numpy as np


class MakeImage(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, kwargs_options, kwargs_data=None, kwargs_psf=None):
        self.Data = Data(kwargs_options, kwargs_data)
        self.LensModel = LensModel(kwargs_options)
        self.SourceModel = SourceModel(kwargs_options)
        self.LensLightModel = LensLightModel(kwargs_options)
        self.PointSource = PointSource(kwargs_options, self.Data)
        self.kwargs_options = kwargs_options
        self.kwargs_psf = kwargs_psf

    def source_surface_brightness(self, kwargs_lens, kwargs_source, kwargs_else, unconvolved=False, de_lensed=False):
        """
        computes the source surface brightness distribution
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: 1d array of surface brightness pixels
        """

        if de_lensed is True:
            x_source, y_source = self.Data.x_grid_sub, self.Data.y_grid_sub
        else:
            x_source, y_source = self.LensModel.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens, kwargs_else)
        source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source)
        source_light_final = self.Data.re_size_convolve(source_light, self.kwargs_psf, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False):
        """
        computes the lens surface brightness distribution
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: 1d array of surface brightness pixels
        """
        lens_light = self.LensLightModel.surface_brightness(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens_light)
        lens_light_final = self.Data.re_size_convolve(lens_light, self.kwargs_psf, unconvolved=unconvolved)
        return lens_light_final

    def image_linear_solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=False):
        """
        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        map_error = self.kwargs_options.get('error_map', False)
        x_source, y_source = self.LensModel.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens, kwargs_else)
        mask = self.Data.mask
        A, error_map = self._response_matrix(self.Data.x_grid_sub, self.Data.y_grid_sub, x_source, y_source, kwargs_source, kwargs_lens_light, kwargs_else, mask, map_error=map_error)
        data = self.Data.data
        d = data*mask
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1/(self.Data.C_D + error_map), d, inv_bool=inv_bool)
        _, _, _, _ = self._update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        return wls_model, error_map, cov_param, param

    def image_with_params(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        """
        make a image with a realisation of linear parameter values "param"
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 1d array of surface brightness pixels of the simulation
        """
        if source_add:
            source_light = self.source_surface_brightness(kwargs_lens, kwargs_source, kwargs_else, unconvolved=unconvolved)
        else:
            source_light = np.zeros_like(self.Data.data)
        if lens_light_add:
            lens_light = self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        else:
            lens_light = np.zeros_like(self.Data.data)
        if point_source_add and self.kwargs_options.get('point_source', False):
            point_source, error_map = self.PointSource.point_source(self.kwargs_psf, kwargs_else)
        else:
            point_source = np.zeros_like(self.Data.data)
            error_map = np.zeros_like(self.Data.data)
        return source_light + lens_light + point_source, error_map

    def _response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_source, kwargs_lens_light, kwargs_else, mask, map_error=False, unconvolved=False):
        kwargs_psf = self.kwargs_psf
        source_light_response, n_source = self.SourceModel.lightModel.functions_split(x_source, y_source, kwargs_source)
        lens_light_response, n_lens_light = self.LensLightModel.lightModel.functions_split(x_grid, y_grid,
                                                                                           kwargs_lens_light)
        n_points = self.PointSource.num_basis(kwargs_psf, kwargs_else)
        num_param = n_points + n_lens_light + n_source

        numPix = self.Data.numData
        A = np.zeros((num_param, numPix))
        if map_error is True:
            error_map = np.zeros(numPix)
        else:
            error_map = 0
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.Data.re_size_convolve(image, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.Data.re_size_convolve(image, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = image
            n += 1
        # response of point sources
        if self.kwargs_options.get('point_source', False):
            A_point, error_map = self.PointSource.point_source_response(kwargs_psf, kwargs_else, map_error=map_error)
            A[n:n+n_points, :] = A_point
            n += n_points
        A = self._add_mask(A, mask)
        return A, error_map

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * mask

    def _update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
        """
        links linear parameters to kwargs arguments
        :param param:
        :return:
        """
        i = 0
        for k, model in enumerate(self.kwargs_options['source_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE','DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_source[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_source[k]['I0_2'] = param[i]
                i += 1
            if model in ['BULDGE_DISK']:
                kwargs_source[k]['I0_b'] = param[i]
                i += 1
                kwargs_source[k]['I0_d'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_source[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_source[k]['amp'] = param[i:i+num_param]
                i += num_param
        for k, model in enumerate(self.kwargs_options['lens_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_lens_light[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_lens_light[k]['I0_2'] = param[i]
                i += 1
            if model in ['BULDGE_DISK']:
                kwargs_source[k]['I0_b'] = param[i]
                i += 1
                kwargs_source[k]['I0_d'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_lens_light[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_lens_light[k]['amp'] = param[i:i+num_param]
                i += num_param
        num_images = self.kwargs_options.get('num_images', 0)
        if num_images > 0 and self.kwargs_options['point_source']:
            if self.kwargs_options.get('fix_magnification', False):
                mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens,
                                                             kwargs_else)
                kwargs_else['point_amp'] = np.abs(mag) * param[i]
                i += 1
            else:
                kwargs_else['point_amp'] = param[i:i+num_images]
                i += num_images
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else
