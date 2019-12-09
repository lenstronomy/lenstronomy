import numpy as np

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import util
from lenstronomy.Util import image_util

from slitronomy.Optimization.sparse_solver import SparseSolverSource
# from slitronomy.Optimization.sparse_solver import SparseSolverSourceLens


class ImageSparseFit(ImageModel):
    """
    #TODO
    linear version class, inherits ImageModel
    """

    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, extinction_class=None, kwargs_numerics={}, likelihood_mask=None,
                 psf_error_map_bool_list=None, kwargs_sparse_solver={}):
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
        super(ImageSparseFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                             source_model_class=source_model_class,
                                             lens_light_model_class=lens_light_model_class,
                                             point_source_class=point_source_class, extinction_class=extinction_class, kwargs_numerics=kwargs_numerics)
        
        source_model_list = self.SourceModel.profile_type_list
        if 'STARLETS' not in source_model_list or len(source_model_list) != 1:
            raise ValueError("'STARLETS' must be the only source model list for sparse fit")

        lens_light_model_list = self.LensLightModel.profile_type_list
        if len(lens_light_model_list) > 0:
            if 'STARLETS' not in lens_light_model_list or len(lens_light_model_list) != 1:
                raise ValueError("'STARLETS' must be the only lens light model list for sparse fit")
            raise NotImplementedError("Sparse optimization for source and lens light not yet implemented in SLITronomy")
            # lens_light_profile_class = 
            # self.sparseSolver = SparseSolverSourceLens(self.Data, self.LensModel, self.SourceModel, self.LensLightModel, psf_class=self.PSF, 
            #                                            likelihood_mask=self.likelihood_mask, **kwargs_sparse_solver)
        else:
            self.sparseSolver = SparseSolverSource(self.Data, self.LensModel, self.SourceModel, psf_class=self.PSF, 
                                                   likelihood_mask=self.likelihood_mask, **kwargs_sparse_solver)
        self._subgrid_res_source = kwargs_sparse_solver.get('subgrid_res_source', 1)


    def source_surface_brightness(self, kwargs_source, kwargs_lens=None, kwargs_extinction=None, kwargs_special=None,
                                  unconvolved=False, de_lensed=False, k=None, re_sized=True, original_grid=True):
        """
        Overwrites ImageModel method.
        ImageModel.source_surface_brightness() may not work for some settings.

        # TODO : make ImageModel.source_surface_brightness() to work without this overwriting.

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_extinction: list of keyword arguments of extinction model
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :param re_sized: returns light distribution on grid with original resolution (if subgrid_res_source > 1)
        :param original_grid: returns light distribution on the original grid (like before reduction to minimal source plane by solver)
        :return: 1d array of surface brightness pixels
        """
        if len(self.SourceModel.profile_type_list) == 0:
            return np.zeros((self.Data.num_pixel_axes))
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(ra_grid, dec_grid, kwargs_source, k=k)
        else:
            # TODO
            raise NotImplementedError
        source_light = util.array2image(source_light)

        if not unconvolved:
            source_light = self.sparseSolver.original_grid_source(source_light)
            source_light = image_util.re_size(source_light, self._subgrid_res_source)
            source_light = self.sparseSolver.psf_convolution(source_light)
        else:
            if original_grid:
                source_light = self.sparseSolver.original_grid_source(source_light)
            if re_sized:
                source_light = image_util.re_size(source_light, self._subgrid_res_source)
        return source_light

    def image_sparse_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
                           kwargs_ps=None, kwargs_extinction=None, kwargs_special=None):
        return self._image_sparse_solve(kwargs_lens, kwargs_source, kwargs_lens_light, 
                                        kwargs_ps, kwargs_extinction, kwargs_special)

    def _image_sparse_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, 
                            kwargs_ps=None, kwargs_extinction=None, kwargs_special=None):
        """
        computes the image (lens and source surface brightness with a given lens model)
        using sparse optimization, on the data pixelated grid.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        C_D_response, model_error = self._error_response(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
        model, param, n_pixels_source = self._solve(kwargs_lens, kwargs_source, kwargs_lens_light)
        cov_param = None
        _, _ = self.update_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, n_pixels_source)
        return model, model_error

    def _solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None):
        """

        :return: 2d numpy array, 3d numpy array
        """
        # solve using sparsity as a prior for surface brightness distributions
        image_model, source_model, _, param = self.sparseSolver.solve(kwargs_lens, kwargs_source, kwargs_lens_light)
        n_pixels_source = source_model.size
        return image_model, param, n_pixels_source

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
        # psf_model_error = self._error_map_psf(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
        # C_D_response = self.image2array_masked(self.Data.C_D + psf_model_error)
        psf_model_error = 0.
        C_D_response = self.image2array_masked(self.Data.C_D)
        return C_D_response, psf_model_error

    def likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                    kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: log likelihood (natural logarithm)
        """
        return self._likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light)


    def _likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :return: log likelihood (natural logarithm)
        """
        # generate image
        im_sim, model_error = self._image_sparse_solve(kwargs_lens, kwargs_source, kwargs_lens_light)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)
        # if cov_matrix is not None and source_marg:
        #     marg_const = de_lens.marginalization_new(cov_matrix, d_prior=linear_prior)
        #     logL += marg_const
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

    def update_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, n_pixels_source=None):
        """

        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        # update amplitudes (wavelets coefficients)
        i = 0
        kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        # update other parameters if required
        if n_pixels_source is not None:
            kwargs_source[0]['n_pixels'] = n_pixels_source
        return kwargs_source, kwargs_lens_light

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
        