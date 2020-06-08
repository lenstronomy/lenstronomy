__author__ = 'aymgal'

from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel


class MultiSparse(object):
    """
    class to simulate/reconstruct images in multi-band option, adapted for sparse (pixel-based) modelling.
    This class calls functions of image_model.py with different bands with
    joint non-linear parameters and decoupled linear parameters.

    the class supports same arguments as MultiLinear class, except that point sources treatment is not supported (yet).
    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, compute_bool=None,
                 kwargs_sparse_solver_list=None):
        """

        :param multi_band_list: list of imaging band configurations [[kwargs_data, kwargs_psf, kwargs_numerics],[...], ...]
        :param kwargs_model: model option keyword arguments
        :param likelihood_mask_list: list of likelihood masks (booleans with size of the individual images
        :param compute_bool: (optinal), bool list to indicate which band to be included in the modeling
        """
        self.type = 'multi-sparse'
        self._num_bands = len(multi_band_list)
        if kwargs_sparse_solver_list is None:
            kwargs_sparse_solver_list = [{}]*self._num_bands
        imageModel_list = []
        for band_index in range(self._num_bands):
            multi_band_type = 'single-band-sparse'
            imageModel = SingleBandMultiModel(multi_band_list, multi_band_type, kwargs_model, likelihood_mask_list=likelihood_mask_list, 
                                              band_index=band_index, kwargs_sparse_solver=kwargs_sparse_solver_list[band_index])
            imageModel_list.append(imageModel)
        if compute_bool is None:
            compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
        self._compute_bool = compute_bool
        self._imageModel_list = imageModel_list
        # self._num_response_list = []
        # for imageModel in imageModel_list:
        #     self._num_response_list.append(imageModel.num_data_evaluate)

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
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
        model_list, error_map_list, param_list = [], [], []
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                model, error_map, param, _ = self._imageModel_list[i].image_linear_solve(kwargs_lens,
                                                                                                     kwargs_source,
                                                                                                     kwargs_lens_light,
                                                                                                     kwargs_ps,
                                                                                                     kwargs_extinction,
                                                                                                     kwargs_special,
                                                                                                     inv_bool=inv_bool)
                model_list.append(model)
                error_map_list.append(error_map)
                param_list.append(param)
        cov_param_list = None
        return model_list, error_map_list, cov_param_list, param_list

    def likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                    kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return: log likelihood (natural logarithm) (sum of the log likelihoods of the individual images)
        """
        # generate image
        logL = 0
        if linear_prior is None:
            linear_prior = [None for i in range(self._num_bands)]
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                logL += self._imageModel_list[i].likelihood_data_given_model(kwargs_lens, kwargs_source,
                                                                             kwargs_lens_light, kwargs_ps,
                                                                             kwargs_extinction, kwargs_special,
                                                                             source_marg=source_marg,
                                                                             linear_prior=linear_prior[i])
        return logL

    @property
    def num_bands(self):
        return self._num_bands

    def reset_point_source_cache(self, bool=True):
        """
        deletes all the cache in the point source class and saves it from then on

        :return:
        """
        # for imageModel in self._imageModel_list:
        #     imageModel.reset_point_source_cache(bool=bool)
        raise NotImplementedError("Point source modelling not tested in multiband in conjuction with sparse modelling.")

    @property
    def num_data_evaluate(self):
        num = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                num += self._imageModel_list[i].num_data_evaluate
        return num

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                num += self._imageModel_list[i].num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        return num

    def reduced_residuals(self, model_list, error_map_list=None):
        """

        :param model_list: list of models
        :param error_map_list: list of error maps
        :return:
        """
        residual_list = []
        if error_map_list is None:
            error_map_list = [[] for i in range(self._num_bands)]
        index = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                residual_list.append(self._imageModel_list[i].reduced_residuals(model_list[index], error_map=error_map_list[index]))
                index += 1
        return residual_list
