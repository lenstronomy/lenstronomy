import numpy as np

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.Util import util
from lenstronomy.Util import image_util
import lenstronomy.ImSim.de_lens as de_lens

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens
from slitronomy.Optimization.solver_source_ps import SparseSolverSourcePS


class ImageSparseFit(ImageLinearFit):
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
        super(ImageSparseFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                             source_model_class=source_model_class,
                                             lens_light_model_class=lens_light_model_class,
                                             point_source_class=point_source_class, extinction_class=extinction_class, 
                                             kwargs_numerics=kwargs_numerics, likelihood_mask=likelihood_mask,
                                             psf_error_map_bool_list=psf_error_map_bool_list)
        
        # TODO : implement support for numba convolution
        # current implementation of lenstronomy does not allow access to the convolution_class through self.ImageNumerics
        # convolution_class = PixelKernelConvolution(self.PSF.kernel_point_source, convolution_type='fft_static')

        no_lens_light = (self.LensLightModel is None or len(self.LensLightModel.profile_type_list) == 0)
        no_point_sources = (self.PointSource is None or len(self.PointSource.point_source_type_list) == 0)
        if no_lens_light and no_point_sources:
            model_list = self.SourceModel.profile_type_list
            if len(model_list) != 1 or model_list[0] not in ['STARLETS', 'STARLETS_GEN2']:
                raise ValueError("'STARLETS' or 'STARLETS_GEN2' must be the only source model list for sparse fit")
            self.sparseSolver = SparseSolverSource(self.Data, self.LensModel, self.SourceModel, self.ImageNumerics, 
                                                   likelihood_mask=likelihood_mask, 
                                                   **kwargs_sparse_solver)
        elif no_point_sources:
            model_list = self.LensLightModel.profile_type_list
            if len(model_list) != 1 or model_list[0] not in ['STARLETS', 'STARLETS_GEN2']:
                raise ValueError("'STARLETS' or 'STARLETS_GEN2' must be the only lens light model list for sparse fit")
            self.sparseSolver = SparseSolverSourceLens(self.Data, self.LensModel, self.SourceModel, self.LensLightModel, self.ImageNumerics, 
                                                       likelihood_mask=likelihood_mask, 
                                                       **kwargs_sparse_solver)
        elif no_lens_light:
            if not np.all(self.PSF.psf_error_map == 0):
                print("WARNING : SparseSolver with point sources does not support PSF error map for now !")
            self.sparseSolver = SparseSolverSourcePS(self.Data, self.LensModel, self.SourceModel, self.ImageNumerics, 
                                                     point_source_linear_solver=self._image_linear_solve_point_sources, #TODO: not fully satisfying
                                                     likelihood_mask=likelihood_mask, 
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

        # TODO : integrate source grid from the sparseSolver into ImageNumerics
        # ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        ra_grid, dec_grid = self.sparseSolver.lensingOperator.sourcePlane.grid()

        source_light = self.SourceModel.surface_brightness(ra_grid, dec_grid, kwargs_source, k=k)
        source_light = util.array2image(source_light)

        if de_lensed is True:
            if not unconvolved:
                # PSF kernel is defined at the original (lower) resolution so image needs to be re-sized
                source_light = self.sparseSolver.project_on_original_grid_source(source_light)

                #TODO: use ImageNumerics like in super class
                source_light = image_util.re_size(source_light, self._subgrid_res_source)
                source_light = self.sparseSolver.psf_convolution(source_light)
            else:
                if original_grid:
                    source_light = self.sparseSolver.project_on_original_grid_source(source_light)
                if re_sized:
                    source_light = image_util.re_size(source_light, self._subgrid_res_source)
        
        else:
            source_light = self.sparseSolver.lensingOperator.source2image_2d(source_light)
            if not unconvolved:
                #TODO: use ImageNumerics like in super class
                source_light = self.sparseSolver.psf_convolution(source_light)

        # TODO : support source grid offsets (using 'delta_x_source_grid' in kwargs_special)
        # i.e. interpolate the image back to center coordinates

        return source_light

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """

        computes the lens surface brightness distribution

        If 'unconvolved' is True, a warning message will appear, and the convolved light is returned.
        This is because the sparse optimizer does not solve for the unconvolved lens light, in order to prevent deconvolutions
        that can otherwise reduce the quality of fit. Hence the deconvolution of the lens light should be performed in post-processing.

        # TODO : make ImageModel.source_surface_brightness() to work without this overwriting.

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: not defined here. Here for keeping same method signatures as in super class.
        :return: 1d array of surface brightness pixels
        """
        if unconvolved is True:
            print("Warning : sparse solver for lens light does not perform deconvolution of lens light, returning convolved estimate instead")
        # ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        ra_grid, dec_grid = self.sparseSolver.lensingOperator.imagePlane.grid()
        lens_light = self.LensLightModel.surface_brightness(ra_grid, dec_grid, kwargs_lens_light, k=k)
        lens_light = util.array2image(lens_light)
        return lens_light

    def image_sparse_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
                           kwargs_ps=None, kwargs_extinction=None, kwargs_special=None):
        #TODO: add the 'inv_bool' parameters like in super.image_linear_solve for point source linear inversion ?
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
        init_ps_model = self.point_source(kwargs_ps, kwargs_lens=kwargs_lens, kwargs_special=kwargs_special)
        model, param, fixed_param = self.sparseSolver.solve(kwargs_lens, kwargs_source,
                                                            kwargs_lens_light=kwargs_lens_light,
                                                            kwargs_ps=kwargs_ps,
                                                            kwargs_special=kwargs_special,
                                                            init_ps_model=init_ps_model)
        cov_param = None
        _, _ = self.update_fixed_kwargs(fixed_param, kwargs_source, kwargs_lens_light)
        _, _, _, _ = self.update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        return model, model_error

    def update_fixed_kwargs(self, fixed_param, kwargs_source, kwargs_lens_light):
        """
        :param param: some parameter vector corresponding for updating kwargs
        :return: updated list of kwargs with linear parameter values
        """
        # TODO : write this method in the same spirit as super().update_linear_kwargs()
        if kwargs_source is not None and len(kwargs_source) > 0:
            n_pixels_source, pixel_scale_source = fixed_param[0], fixed_param[1]
            kwargs_source[0]['n_pixels'] = n_pixels_source
            kwargs_source[0]['scale'] = pixel_scale_source
            kwargs_source[0]['center_x'] = 0
            kwargs_source[0]['center_y'] = 0
        if kwargs_lens_light is not None and len(kwargs_lens_light) > 0:
            n_pixels_lens_light, pixel_scale_lens_light = fixed_param[2], fixed_param[3]
            kwargs_lens_light[0]['n_pixels'] = n_pixels_source
            kwargs_lens_light[0]['scale'] = pixel_scale_source
            kwargs_lens_light[0]['center_x'] = 0
            kwargs_lens_light[0]['center_y'] = 0
        return kwargs_source, kwargs_lens_light

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
        im_sim, model_error = self._image_sparse_solve(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, 
                                                       kwargs_lens_light=kwargs_lens_light, kwargs_special=kwargs_special)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)
        if not np.isfinite(logL):
            return -1e20  # penalty
        # if cov_matrix is not None and source_marg:
        #     marg_const = de_lens.marginalization_new(cov_matrix, d_prior=linear_prior)
        #     logL += marg_const
        return logL

    def _image_linear_solve_point_sources(self, sparse_model, kwargs_lens=None, kwargs_ps=None, kwargs_special=None, inv_bool=False):
        """

        linear solve, but only for point sources. The target image is the imaging data with sparse model subtracted (source and lens light)

        computes the image (point source amplitudes with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        A = self._linear_response_point_sources(kwargs_lens, kwargs_ps, kwargs_special)
        C_D_response, model_error = self._error_response(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
        d = self.data_response - sparse_model  # subract source light + lens light model
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _ = self.update_linear_kwargs_point_sources(param, kwargs_lens, kwargs_ps)
        model = self.array_masked2image(wls_model)
        return model, model_error, cov_param, param

    def _linear_response_point_sources(self, kwargs_lens, kwargs_ps, kwargs_special):
        """

        return linear response Matrix, with only point sources.

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param unconvolved:
        :return:
        """
        ra_pos, dec_pos, amp, num_param = self.point_source_linear_response_set(kwargs_ps, kwargs_lens, kwargs_special, with_amp=False)
        num_response = self.num_data_evaluate
        A = np.zeros((num_param, num_response))
        # response of point sources
        for i in range(0, num_param):
            image = self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
            A[i, :] = self.image2array_masked(image)
        return np.nan_to_num(A)
        
    def update_linear_kwargs_point_sources(self, param, kwargs_lens, kwargs_ps):
        """

        links linear parameters to kwargs arguments, with only point sources.

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_ps
