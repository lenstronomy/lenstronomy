import numpy as np

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import util
from lenstronomy.Util import image_util

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens


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
        
        # TODO : implement support for numba convolution
        # current implementation of lenstronomy does not allow access to the convolution_class through self.ImageNumerics
        convolution_class = PixelKernelConvolution(self.PSF.kernel_point_source, convolution_type='fft_static')

        source_model_list = self.SourceModel.profile_type_list
        if 'STARLETS' not in source_model_list or len(source_model_list) != 1:
            raise ValueError("'STARLETS' must be the only source model list for sparse fit")

        lens_light_model_list = self.LensLightModel.profile_type_list
        if len(lens_light_model_list) > 0:
            if 'STARLETS' not in lens_light_model_list or len(lens_light_model_list) != 1:
                raise ValueError("'STARLETS' must be the only lens light model list for sparse fit")
            self.sparseSolver = SparseSolverSourceLens(self.Data, self.LensModel, self.SourceModel, self.LensLightModel, psf_class=self.PSF, 
                                                       convolution_class=convolution_class, likelihood_mask=self.likelihood_mask, 
                                                       **kwargs_sparse_solver)
        else:
            self.sparseSolver = SparseSolverSource(self.Data, self.LensModel, self.SourceModel, psf_class=self.PSF, 
                                                   convolution_class=convolution_class, likelihood_mask=self.likelihood_mask, 
                                                   **kwargs_sparse_solver)
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
            # PSF kernel is defined at the original (lower) resolution so image needs to be re-sized
            source_light = self.sparseSolver.project_original_grid_source(source_light)
            source_light = image_util.re_size(source_light, self._subgrid_res_source)
            source_light = self.sparseSolver.psf_convolution(source_light)
        else:
            if original_grid:
                source_light = self.sparseSolver.project_original_grid_source(source_light)
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
        model, param, fixed_param = self._solve(kwargs_lens, kwargs_source, kwargs_lens_light)
        cov_param = None
        _, _, _, _ = self.update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        _, _ = self.update_fixed_kwargs(fixed_param, kwargs_source, kwargs_lens_light)
        return model, model_error

    def _solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None):
        """

        :return: 2d numpy array, 3d numpy array
        """
        # solve using sparsity as a prior for surface brightness distributions
        image_model, source_model, lens_light_model, param = self.sparseSolver.solve(kwargs_lens, kwargs_source, 
                                                                                     kwargs_lens_light=kwargs_lens_light)
        fixed_param = [source_model.size]
        if lens_light_model is not None:
            fixed_param.append(lens_light_model.size)
        else:
            fixed_param.append(None)
        return image_model, param, fixed_param

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

    def _likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                     kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None):
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

    def update_fixed_kwargs(self, fixed_param, kwargs_source, kwargs_lens_light):
        """
        :param param: some parameter vector corresponding for updating kwargs
        :return: updated list of kwargs with linear parameter values
        """
        # TODO : write this method in the spirit than super().update_linear_kwargs() 
        if fixed_param[0] is not None:
            n_pixels_source = fixed_param[0]
            kwargs_source[0]['n_pixels'] = n_pixels_source
        if fixed_param[1] is not None:
            n_pixels_lens_light = fixed_param[1]
            kwargs_lens_light[0]['n_pixels'] = n_pixels_lens_light
        return kwargs_source, kwargs_lens_light

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
        