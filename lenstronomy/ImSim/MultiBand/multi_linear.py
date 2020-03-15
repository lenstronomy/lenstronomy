from lenstronomy.ImSim.MultiBand.multi_data_base import MultiDataBase
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel


class MultiLinear(MultiDataBase):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    joint non-linear parameters and decoupled linear parameters.
    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, compute_bool=None):
        self.type = 'multi-linear'
        imageModel_list = []
        for band_index in range(len(multi_band_list)):
            multi_band_type = 'single-band'  #TODO: (aymgal) support 'single-band-sparse'
            imageModel = SingleBandMultiModel(multi_band_list, multi_band_type, kwargs_model, likelihood_mask_list=likelihood_mask_list, 
                                              band_index=band_index)
            imageModel_list.append(imageModel)
        super(MultiLinear, self).__init__(imageModel_list, compute_bool=compute_bool)

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
                wls_list.append(wls_model)
                error_map_list.append(error_map)
                cov_param_list.append(cov_param)
                param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

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
