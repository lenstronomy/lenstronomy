from lenstronomy.ImSim.MultiBand.multi_data_base import MultiDataBase
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

__all__ = ['MultiLinear']


class MultiLinear(MultiDataBase):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    joint non-linear parameters and decoupled linear parameters.

    the class supports keyword arguments 'index_lens_model_list', 'index_source_light_model_list',
    'index_lens_light_model_list', 'index_point_source_model_list', 'index_optical_depth_model_list' in kwargs_model
    These arguments should be lists of length the number of imaging bands available and each entry in the list
    is a list of integers specifying the model components being evaluated for the specific band.

    E.g. there are two bands and you want to different light profiles being modeled.
    - you define two different light profiles lens_light_model_list = ['SERSIC', 'SERSIC']
    - set index_lens_light_model_list = [[0], [1]]
    - (optional) for now all the parameters between the two light profiles are independent in the model. You have
    the possibility to join a subset of model parameters (e.g. joint centroid). See the Param() class for documentation.

    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, compute_bool=None, kwargs_pixelbased=None):
        """

        :param multi_band_list: list of imaging band configurations [[kwargs_data, kwargs_psf, kwargs_numerics],[...], ...]
        :param kwargs_model: model option keyword arguments
        :param likelihood_mask_list: list of likelihood masks (booleans with size of the individual images
        :param compute_bool: (optinal), bool list to indicate which band to be included in the modeling
        """
        self.type = 'multi-linear'
        imageModel_list = []
        for band_index in range(len(multi_band_list)):
            imageModel = SingleBandMultiModel(multi_band_list, kwargs_model, likelihood_mask_list=likelihood_mask_list,
                                              band_index=band_index, kwargs_pixelbased=kwargs_pixelbased)
            imageModel_list.append(imageModel)
        super(MultiLinear, self).__init__(imageModel_list, compute_bool=compute_bool)

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           kwargs_extinction=None, kwargs_special=None, inv_bool=False):
        """
        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization
        (i.e. flux normalization of the brightness profiles)

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        wls_list, error_map_list, cov_param_list, param_list = [], [], [], []
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                wls_model, error_map, cov_param, param = self._imageModel_list[i].image_linear_solve(kwargs_lens,
                                                                                                     kwargs_source,
                                                                                                     kwargs_lens_light,
                                                                                                     kwargs_ps,
                                                                                                     kwargs_extinction,
                                                                                                     kwargs_special,
                                                                                                     inv_bool=inv_bool)
            else:
                wls_model, error_map, cov_param, param = None, None, None, None
            wls_list.append(wls_model)
            error_map_list.append(error_map)
            cov_param_list.append(cov_param)
            param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

    def likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                    kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None,
                                    check_positive_flux=False):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param check_positive_flux: bool, if True, checks whether the linear inversion resulted in non-negative flux
         components and applies a punishment in the likelihood if so.
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
                                                                             linear_prior=linear_prior[i],
                                                                             check_positive_flux=check_positive_flux)
        return logL
