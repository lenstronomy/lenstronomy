from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.Util import util
import numpy as np


class ImageFit(ImageModel):
    """
    image fit class, to be generalized through 'linear' or 'sparse' inversions, inherits ImageModel
    """
    def __init__(self, data_class, psf_class, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, extinction_class=None, kwargs_numerics={}, likelihood_mask=None,
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
        #kwargs_numerics['compute_indexes'] = self.likelihood_mask  # here we overwrite the indexes to be computed with the likelihood mask
        super(ImageFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                       source_model_class=source_model_class, lens_light_model_class=lens_light_model_class,
                                       point_source_class=point_source_class, extinction_class=extinction_class, 
                                       kwargs_numerics=kwargs_numerics)
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(self.PointSource.point_source_type_list)
        self._psf_error_map_bool_list = psf_error_map_bool_list

    @property
    def data_response(self):
        """
        returns the 1d array of the data element that is fitted for (including masking)

        :return: 1d numpy array
        """
        d = self.image2array_masked(self.Data.data)
        return d

    def error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        return self._error_response(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)

    def _error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        psf_model_error = self._error_map_psf(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
        C_D_response = self.image2array_masked(self.Data.C_D + psf_model_error)
        return C_D_response, psf_model_error

    def reduced_residuals(self, model, error_map=0):
        """

        :param model: 2d numpy array of the modeled image
        :param error_map: 2d numpy array of additional noise/error terms from model components (such as PSF model uncertainties)
        :return: 2d numpy array of reduced residuals per pixel
        """
        mask = self.likelihood_mask
        C_D = self.Data.C_D_model(model)
        residual = (model - self.Data.data)/np.sqrt(C_D+np.abs(error_map))*mask
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

    def _error_map_psf(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        error_map = np.zeros((self.Data.num_pixel_axes))
        if self._psf_error_map is True:
            for k, bool in enumerate(self._psf_error_map_bool_list):
                if bool is True:
                    ra_pos, dec_pos, amp = self.PointSource.point_source_list(kwargs_ps, kwargs_lens=kwargs_lens, k=k)
                    if len(ra_pos) > 0:
                        ra_pos, dec_pos = self._displace_astrometry(ra_pos, dec_pos, kwargs_special=kwargs_special)
                        error_map += self.ImageNumerics.psf_error_map(ra_pos, dec_pos, amp, self.Data.data)
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

    def point_source_linear_response_set(self, kwargs_ps, kwargs_lens, kwargs_special, with_amp=True):
        """

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param kwargs_special: special keyword argument list, may include 'delta_x_image' and 'delta_y_image'
        :param with_amp: bool, if True, relative magnification between multiply imaged point sources are held fixed.
        :return: list of positions and amplitudes split in different basis components with applied astrometric corrections
        """

        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=with_amp)

        if kwargs_special is not None:
            if 'delta_x_image' in kwargs_special:
                delta_x, delta_y = kwargs_special['delta_x_image'], kwargs_special['delta_y_image']
                k = 0
                n = len(delta_x)
                for i in range(n_points):
                    for j in range(len(ra_pos[i])):
                        if k >= n:
                            break
                        ra_pos[i][j] = ra_pos[i][j] + delta_x[k]
                        dec_pos[i][j] = dec_pos[i][j] + delta_y[k]
                        k += 1
        return ra_pos, dec_pos, amp, n_points
        