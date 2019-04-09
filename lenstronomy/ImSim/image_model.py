__author__ = 'sibirrer'

from lenstronomy.ImSim.image_numerics import ImageNumerics
import lenstronomy.ImSim.de_lens as de_lens
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
        self.ImageNumerics = ImageNumerics(data=self.Data, psf=self.PSF, **kwargs_numerics)
        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class
        self.PointSource = point_source_class
        self._error_map_bool_list = None
        if self.PointSource is not None:
            self.PointSource.update_lens_model(lens_model_class=lens_model_class)
            x_center, y_center = self.Data.center
            self.PointSource.update_search_window(search_window=self.Data.width, x_center=x_center, y_center=y_center)
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
        self.ImageNumerics._Data = data_class

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

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, inv_bool=False):
        """

        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        A = self.linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        C_D_response, model_error = self.error_response(kwargs_lens, kwargs_ps)
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _, _, _ = self._update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.ImageNumerics.array2image(wls_model)
        return model, model_error, cov_param, param

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        A = self._response_matrix(self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting,
                                      kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, self.ImageNumerics.mask)
        return A

    @property
    def data_response(self):
        """
        returns the 1d array of the data element that is fitted for (including masking)

        :return: 1d numpy array
        """
        d = self.ImageNumerics.image2array(self.Data.data * self.ImageNumerics.mask)
        return d

    def error_response(self, kwargs_lens, kwargs_ps):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        model_error = self.error_map(kwargs_lens, kwargs_ps)
        error_map_1d = self.ImageNumerics.image2array(model_error)
        C_D_response = self.ImageNumerics.C_D_response + error_map_1d
        return C_D_response, model_error

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
                        error_map_add = self.ImageNumerics.psf_error_map(ra_pos[i], dec_pos[i], amp[i])
                        error_map += error_map_add
        return error_map

    def point_sources_list(self, kwargs_ps, kwargs_lens, k=None):
        """

        :param kwargs_ps:
        :return: list of images containing only single point sources
        """
        point_list = []
        if self.PointSource is None:
            return point_list
        ra_array, dec_array, amp_array = self.PointSource.point_source_list(kwargs_ps, kwargs_lens, k=k)
        for i in range(len(ra_array)):
            point_source = self.ImageNumerics.point_source_rendering([ra_array[i]], [dec_array[i]], [amp_array[i]])
            point_list.append(point_source)
        return point_list

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg=False):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: log likelihood (natural logarithm)
        """
        # generate image
        im_sim, model_error, cov_matrix, param = self.image_linear_solve(kwargs_lens, kwargs_source,
                                                                         kwargs_lens_light, kwargs_ps,
                                                                         inv_bool=source_marg)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.ImageNumerics.mask, model_error)
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalisation_const(cov_matrix)
            #if marg_const + logL > 0:
            #logL = np.log(np.exp(logL) + np.exp(marg_const))
            logL += marg_const
        return logL

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

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        num += self.SourceModel.num_param_linear(kwargs_source)
        num += self.LensLightModel.num_param_linear(kwargs_lens_light)
        num += self.PointSource.num_basis(kwargs_ps, kwargs_lens)
        return num

    def _response_matrix(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, mask, unconvolved=False):
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
            source_light_response, n_source = self.source_mapping.image_flux_split(x_grid, y_grid, kwargs_lens,
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
        return np.nan_to_num(A)

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * self.ImageNumerics.image2array(mask)

    def _update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        if self.SourceModel is not None:
            kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        if self.LensLightModel is not None:
            kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        if self.PointSource is not None:
            kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
