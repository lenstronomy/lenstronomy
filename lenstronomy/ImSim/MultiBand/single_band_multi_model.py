from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.ImSim.image_sparse_solve import ImageSparseFit
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import class_creator


class SingleBandMultiModel(ImageLinearFit):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    decoupled linear parameters and the option to pass/select different light models for the different bands

    the class instance needs to have a forth row in the multi_band_list with keyword arguments 'source_light_model_index' and
    'lens_light_model_index' as bool arrays of the size of the total source model types and lens light model types,
    specifying which model is evaluated for which band.

    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, band_index=0):
        self.type = 'single-band-multi-model'
        if likelihood_mask_list is None:
            likelihood_mask_list = [None for i in range(len(multi_band_list))]
        lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class = class_creator.create_class_instances(band_index=band_index, **kwargs_model)
        kwargs_data = multi_band_list[band_index][0]
        kwargs_psf = multi_band_list[band_index][1]
        kwargs_numerics = multi_band_list[band_index][2]
        data_i = ImageData(**kwargs_data)
        psf_i = PSF(**kwargs_psf)

        index_lens_model_list = kwargs_model.get('index_lens_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_model = index_lens_model_list[band_index]
        index_source_list = kwargs_model.get('index_source_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_source = index_source_list[band_index]
        index_lens_light_list = kwargs_model.get('index_lens_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_light = index_lens_light_list[band_index]
        index_point_source_list = kwargs_model.get('index_point_source_model_list', [None for i in range(len(multi_band_list))])
        self._index_point_source = index_point_source_list[band_index]
        index_optical_depth = kwargs_model.get('index_optical_depth_model_list',
                                                   [None for i in range(len(multi_band_list))])
        self._index_optical_depth = index_optical_depth[band_index]

        super(SingleBandMultiModel, self).__init__(data_i, psf_i, lens_model_class, source_model_class,
                                                   lens_light_model_class, point_source_class, extinction_class,
                                                   kwargs_numerics=kwargs_numerics, likelihood_mask=likelihood_mask_list[band_index])

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
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        wls_model, error_map, cov_param, param = self._image_linear_solve(kwargs_lens_i, kwargs_source_i,
                                                                          kwargs_lens_light_i, kwargs_ps_i,
                                                                          kwargs_extinction_i, kwargs_special, inv_bool=inv_bool)
        return wls_model, error_map, cov_param, param

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
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        logL = self._likelihood_data_given_model(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i,
                                                 kwargs_extinction_i, kwargs_special, source_marg=source_marg,
                                                 linear_prior=linear_prior)
        return logL

    def num_param_linear(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        num = self._num_param_linear(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)
        return num

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                               kwargs_extinction=None, kwargs_special=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        A = self._linear_response_matrix(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i,
                                         kwargs_extinction_i, kwargs_special)
        return A

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
        if self._index_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        return self._error_map_source(kwargs_source_i, x_grid, y_grid, cov_param)

    def select_kwargs(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                      kwargs_extinction=None, kwargs_special=None):
        """
        select subset of kwargs lists referenced to this imaging band

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        if self._index_lens_model is None:
            kwargs_lens_i = kwargs_lens
        else:
            kwargs_lens_i = [kwargs_lens[k] for k in self._index_lens_model]
        if self._index_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        if self._index_lens_light is None:
            kwargs_lens_light_i = kwargs_lens_light
        else:
            kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light]
        if self._index_point_source is None:
            kwargs_ps_i = kwargs_ps
        else:
            kwargs_ps_i = [kwargs_ps[k] for k in self._index_point_source]
        if self._index_optical_depth is None or kwargs_extinction is None:
            kwargs_extinction_i = kwargs_extinction
        else:
            kwargs_extinction_i = [kwargs_extinction[k] for k in self._index_optical_depth]
        return kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i


class SingleBandMultiModelSparse(ImageSparseFit):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    decoupled linear parameters and the option to pass/select different light models for the different bands

    the class instance needs to have a forth row in the multi_band_list with keyword arguments 'source_light_model_index' and
    'lens_light_model_index' as bool arrays of the size of the total source model types and lens light model types,
    specifying which model is evaluated for which band.

    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, band_index=0, kwargs_sparse_solver={}):
        self.type = 'single-band-multi-model-sparse'
        if likelihood_mask_list is None:
            likelihood_mask_list = [None for i in range(len(multi_band_list))]
        lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class = class_creator.create_class_instances(band_index=band_index, **kwargs_model)
        kwargs_data = multi_band_list[band_index][0]
        kwargs_psf = multi_band_list[band_index][1]
        kwargs_numerics = multi_band_list[band_index][2]
        data_i = ImageData(**kwargs_data)
        psf_i = PSF(**kwargs_psf)

        index_lens_model_list = kwargs_model.get('index_lens_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_model = index_lens_model_list[band_index]
        index_source_list = kwargs_model.get('index_source_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_source = index_source_list[band_index]
        index_lens_light_list = kwargs_model.get('index_lens_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_light = index_lens_light_list[band_index]
        index_point_source_list = kwargs_model.get('index_point_source_model_list', [None for i in range(len(multi_band_list))])
        self._index_point_source = index_point_source_list[band_index]
        index_optical_depth = kwargs_model.get('index_optical_depth_model_list',
                                                   [None for i in range(len(multi_band_list))])
        self._index_optical_depth = index_optical_depth[band_index]

        super(SingleBandMultiModelSparse, self).__init__(data_i, psf_i, lens_model_class, source_model_class,
                                                   lens_light_model_class, point_source_class, extinction_class,
                                                   kwargs_numerics=kwargs_numerics, likelihood_mask=likelihood_mask_list[band_index],
                                                   kwargs_sparse_solver=kwargs_sparse_solver)

    def image_sparse_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
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
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        wls_model, error_map = self._image_sparse_solve(kwargs_lens_i, kwargs_source_i,
                                                        kwargs_lens_light_i)
        return wls_model, error_map

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
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        logL = self._likelihood_data_given_model(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i)
        return logL

    def num_param_linear(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        num = self._num_param_linear(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)
        return num

    def select_kwargs(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                      kwargs_extinction=None, kwargs_special=None):
        """
        select subset of kwargs lists referenced to this imaging band

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        if self._index_lens_model is None:
            kwargs_lens_i = kwargs_lens
        else:
            kwargs_lens_i = [kwargs_lens[k] for k in self._index_lens_model]
        if self._index_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        if self._index_lens_light is None:
            kwargs_lens_light_i = kwargs_lens_light
        else:
            kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light]
        if self._index_point_source is None:
            kwargs_ps_i = kwargs_ps
        else:
            kwargs_ps_i = [kwargs_ps[k] for k in self._index_point_source]
        if self._index_optical_depth is None or kwargs_extinction is None:
            kwargs_extinction_i = kwargs_extinction
        else:
            kwargs_extinction_i = [kwargs_extinction[k] for k in self._index_optical_depth]
        return kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i
