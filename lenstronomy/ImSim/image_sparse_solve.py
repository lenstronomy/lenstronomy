from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


class ImageSparseFit(ImageLinearFit):
    """
    #TODO
    linear version class, inherits ImageModel
    """

    def __init__(self, data_class, sparse_optimizer_class, psf_class=None, lens_model_class=None, source_model_class=None,
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
        self._sparse_optimizer_class = sparse_optimizer_class
        super(ImageLinearFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class, 
                                             source_model_class=source_model_class, lens_light_model_class=lens_light_model_class, 
                                             point_source_class=point_source_class, extinction_class=extinction_class, 
                                             kwargs_numerics=kwargs_numerics, likelihood_mask=likelihood_mask,
                                             psf_error_map_bool_list=psf_error_map_bool_list)
        source_model_list = self.SourceModel.profile_type_list
        if 'STARLETS' not in source_model_list:
            raise ValueError("'STARLETS' must be in source model list for sparse fit")
        if len(source_model_list) == 1:
            source_light_response = np.zeros_like(x_grid)
            n_source = 0
            return source_light_response, n_source
        else:
            raise ValueError("'STARLETS' model can not be used with other source light models")


    def _image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                            kwargs_extinction=None, kwargs_special=None, inv_bool=False):
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
        A = self._linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction, kwargs_special)
        C_D_response, model_error = self._error_response(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
        d = self.data_response - self.pixel_surface_brightness(kwargs_lens)
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _, _, _ = self.update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.array_masked2image(wls_model)
        return model, model_error, cov_param, param


    def pixel_surface_brightness(self, kwargs_lens):
        """

        :return: 1d numpy array
        """
        threshold = 5
        lensing_operator = 
        return self._solve_sparse(lensing_operator, threshold)


    def _solve_sparse(self, lensing_operator, k_max, n_iter, size, PSF, PSFconj, S0 = [0], levels = [0], scheme = 'FB',
                      mask = [0], lvl = 0, weightS = 1, noise = 'gaussian', tau = 0, verbosity = 0, nweights = 1,
                      save_steps_dir=None, show_plots=False):
        """SLIT algorithm"""


    def _source_linear_response_matrix(self, x_grid, y_grid, kwargs_lens, kwargs_source):
        """

        return linear response Matrix for source light only
        In this particular case, pixel-base profile is assumed so a blank image is returned

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param unconvolved:
        :return:
        """
        source_light_response = np.zeros_like(x_grid)
        n_source = 0
        return source_light_response, n_source