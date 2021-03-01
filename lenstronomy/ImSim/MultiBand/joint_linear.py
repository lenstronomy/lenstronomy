from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear
import lenstronomy.ImSim.de_lens as de_lens

import numpy as np

__all__ = ['JointLinear']


class JointLinear(MultiLinear):
    """
    class to model multiple exposures in the same band and makes a constraint fit to all bands simultaneously
    with joint constraints on the surface brightness of the model. This model setting require the same surface
    brightness models to be called in all available images/bands

    """
    def __init__(self, multi_band_list, kwargs_model, compute_bool=None, likelihood_mask_list=None):
        #TODO make this raise statement valid
        #if kwargs_model.get('index_source_light_model_list', None) is not None or \
        #        kwargs_model.get('index_lens_light_model_list', None) is not None or \
        #        kwargs_model.get('index_point_source_model_list', None) is not None:
        #    raise ValueError('You are not allowed to set partial surface brightness models to individual bands in the '
        #                     'joint-linear mode. Your settings are: ', kwargs_model)
        super(JointLinear, self).__init__(multi_band_list, kwargs_model=kwargs_model, compute_bool=compute_bool,
                                          likelihood_mask_list=likelihood_mask_list)
        self.type = 'joint-linear'

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
        A = self.linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction, kwargs_special)
        C_D_response, model_error_list = self.error_response(kwargs_lens, kwargs_ps)
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        wls_list = self._array2image_list(wls_model)
        return wls_list, model_error_list, cov_param, param

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                               kwargs_extinction=None, kwargs_special=None):
        """
        computes the linear response matrix (m x n), with n being the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        A = []
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                A_i = self._imageModel_list[i].linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light,
                                                                      kwargs_ps, kwargs_extinction, kwargs_special)
                if len(A) == 0:
                    A = A_i
                else:
                    A = np.append(A, A_i, axis=1)
        return A

    @property
    def data_response(self):
        """
        returns the 1d array of the data element that is fitted for (including masking)

        :return: 1d numpy array
        """
        d = []
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                d_i = self._imageModel_list[i].data_response
                if len(d) == 0:
                    d = d_i
                else:
                    d = np.append(d, d_i)
        return d

    def _array2image_list(self, array):
        """
        maps 1d vector of joint exposures in list of 2d images of single exposures

        :param array: 1d numpy array
        :return: list of 2d numpy arrays of size  of exposures
        """
        image_list = []
        k = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                num_data = self.num_response_list[i]
                array_i = array[k:k + num_data]
                image_i = self._imageModel_list[i].array_masked2image(array_i)
                image_list.append(image_i)
                k += num_data
        return image_list

    def error_response(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        C_D_response, model_error = [], []
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                C_D_response_i, model_error_i = self._imageModel_list[i].error_response(kwargs_lens, kwargs_ps, kwargs_special=kwargs_special)
                model_error.append(model_error_i)
                if len(C_D_response) == 0:
                    C_D_response = C_D_response_i
                else:
                    C_D_response = np.append(C_D_response, C_D_response_i)
        return C_D_response, model_error

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
        im_sim_list, model_error_list, cov_matrix, param = self.image_linear_solve(kwargs_lens, kwargs_source,
                                                                                   kwargs_lens_light, kwargs_ps,
                                                                                   kwargs_extinction, kwargs_special,
                                                                                   inv_bool=source_marg)
        # compute X^2
        logL = 0
        index = 0
        for i in range(self._num_bands):
            if self._compute_bool[i] is True:
                logL += self._imageModel_list[i].Data.log_likelihood(im_sim_list[index], self._imageModel_list[i].likelihood_mask, model_error_list[index])
                index += 1
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalization_new(cov_matrix, d_prior=linear_prior)
            logL += marg_const
        if check_positive_flux is True and self._num_bands > 0:
            bool = self._imageModel_list[0].check_positive_flux(kwargs_source, kwargs_lens_light, kwargs_ps)
            if bool is False:
                logL -= 10 ** 5
        return logL
