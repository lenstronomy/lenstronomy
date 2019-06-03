from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.Util import util
import numpy as np


class ImageLinearFit(ImageModel):
    """
    linear version class, inherits ImageModel
    """
    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, kwargs_numerics={}, likelihood_mask=None,
                 psf_error_map_bool_list=None):
        """

        :param data_class: ImageData() instance
        :param psf_class: PSF() instance
        :param lens_model_class: LensModel() instance
        :param source_model_class: LightModel() instance
        :param lens_light_model_class: LightModel() instance
        :param point_source_class: PointSource() instance
        :param kwargs_numerics: keyword arguments passed to the Numerics module
        :param likelihood_mask: 2d boolean array of pixels to be counted in the likelihood calculation/linear optimization
        :param psf_error_map_bool_list: list of boolean of length of point source models. Indicates whether PSF error map
        being applied to the point sources.
        """
        if likelihood_mask is None:
            likelihood_mask = np.ones_like(data_class.data)
        self.likelihood_mask = np.array(likelihood_mask, dtype=bool)
        self._mask1d = util.image2array(self.likelihood_mask)
        kwargs_numerics['compute_indexes'] = self.likelihood_mask  # here we overwrite the indexes to be computed with the likelihood mask
        super(ImageLinearFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                             source_model_class=source_model_class,
                                             lens_light_model_class=lens_light_model_class,
                                             point_source_class=point_source_class, kwargs_numerics=kwargs_numerics)
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(self.PointSource.point_source_type_list)
        self._psf_error_map_bool_list = psf_error_map_bool_list

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           inv_bool=False):
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
        return self._image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    def _image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           inv_bool=False):
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
        A = self._linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        C_D_response, model_error = self._error_response(kwargs_lens, kwargs_ps)
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _, _, _ = self.update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.array_masked2image(wls_model)
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
        A = self._linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        return A

    @property
    def data_response(self):
        """
        returns the 1d array of the data element that is fitted for (including masking)

        :return: 1d numpy array
        """
        d = self.image2array_masked(self.Data.data)
        return d

    def error_response(self, kwargs_lens, kwargs_ps):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        return self._error_response(kwargs_lens, kwargs_ps)

    def _error_response(self, kwargs_lens, kwargs_ps):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        psf_model_error = self._error_map_psf(kwargs_lens, kwargs_ps)
        C_D_response = self.image2array_masked(self.Data.C_D + psf_model_error)
        return C_D_response, psf_model_error

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
        return self._likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg)

    def _likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg=False):
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
        im_sim, model_error, cov_matrix, param = self._image_linear_solve(kwargs_lens, kwargs_source,
                                                                         kwargs_lens_light, kwargs_ps,
                                                                         inv_bool=source_marg)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalisation_const(cov_matrix)
            # if marg_const + logL > 0:
            # logL = np.log(np.exp(logL) + np.exp(marg_const))
            logL += marg_const
        return logL

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        return self._num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    def _num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        num += self.SourceModel.num_param_linear(kwargs_source)
        num += self.LensLightModel.num_param_linear(kwargs_lens_light)
        num += self.PointSource.num_basis(kwargs_ps, kwargs_lens)
        return num

    def _linear_response_matrix(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                unconvolved=False):
        """

        return linear response Matrix

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param unconvolved:
        :return:
        """
        x_grid, y_grid = self.ImageNumerics.coordinates_evaluate
        source_light_response, n_source = self.source_mapping.image_flux_split(x_grid, y_grid, kwargs_lens,
                                                                               kwargs_source)
        lens_light_response, n_lens_light = self.LensLightModel.functions_split(x_grid, y_grid, kwargs_lens_light)
        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=False)
        num_param = n_points + n_lens_light + n_source

        num_response = self.num_data_evaluate
        A = np.zeros((num_param, num_response))
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.image2array_masked(image)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.image2array_masked(image)
            n += 1
        # response of point sources
        for i in range(0, n_points):
            image = self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
            A[n, :] = self.image2array_masked(image)
            n += 1
        return np.nan_to_num(A)

    def update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def reduced_residuals(self, model, error_map=0):
        """

        :param model:
        :return:
        """
        mask = self.likelihood_mask
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
        return np.sum(chi2**2) / self.num_data_evaluate

    @property
    def num_data_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return int(np.sum(self.likelihood_mask))

    def update_data(self, data_class):
        """

        :param data_class: instance of Data() class
        :return: no return. Class is updated.
        """
        self.Data = data_class
        self.ImageNumerics._PixelGrid = data_class

    def image2array_masked(self, image):
        """
        returns 1d array of values in image that are not masked out for the likelihood computation/linear minimization
        :param image: 2d numpy array of full image
        :return: 1d array
        """
        array = util.image2array(image)
        return array[self._mask1d]

    def array_masked2image(self, array):
        """

        :param array: 1d array of values not masked out (part of linear fitting)
        :return: 2d array of full image
        """
        nx, ny = self.Data.num_pixel_axes
        grid1d = np.zeros(nx * ny)
        grid1d[self._mask1d] = array
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    def _error_map_psf(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        error_map = np.zeros((self.Data.num_pixel_axes))
        if self._psf_error_map is True:
            for k, bool in enumerate(self._psf_error_map_bool_list):
                if bool is True:
                    ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, k=k)
                    for i in range(0, n_points):
                        error_map_add = self.ImageNumerics.psf_error_map(ra_pos[i], dec_pos[i], amp[i], self.Data.data)
                        error_map += error_map_add
        return error_map

    def error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """
        variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source reconstruction as a sum of the errors
        of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """
        return self._error_map_source(kwargs_source, x_grid, y_grid, cov_param)

    def _error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """
        variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source reconstruction as a sum of the errors
        of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """

        error_map = np.zeros_like(x_grid)
        basis_functions, n_source = self.SourceModel.functions_split(x_grid, y_grid, kwargs_source)
        basis_functions = np.array(basis_functions)

        if cov_param is not None:
            for i in range(len(error_map)):
                error_map[i] = basis_functions[:, i].T.dot(cov_param[:n_source, :n_source]).dot(basis_functions[:, i])
        return error_map