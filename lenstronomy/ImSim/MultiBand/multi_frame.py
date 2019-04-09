from lenstronomy.ImSim.MultiBand.multi_data_base import MultiDataBase
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
import lenstronomy.ImSim.de_lens as de_lens

import copy
import numpy as np


class MultiFrame(MultiDataBase):
    """
    class to model multiple patches of the sky simultaneous (e.g. multiple images in a cluster) with different lens models
    for each frame but with shared light components (source and lens)

    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics, kwargs_index], [...], ...]
    kwargs_index: supports:
     'idex_lens_model_list': [0, 1, 2]

    """
    def __init__(self, multi_band_list, lens_model_list=None, source_model_class=None, lens_light_model_class=None,
                 point_source_class=None, compute_bool=None):
        self.type = 'multi-frame'
        if compute_bool is None:
            compute_bool = [True] * len(multi_band_list)
        else:
            if not len(compute_bool) == len(multi_band_list):
                raise ValueError('compute_bool statement has not the same range as number of bands available! (%s vs %s)' % (len(compute_bool), len(multi_band_list)))
        imageModel_list = []
        self._idex_lens_list = []
        for i in range(len(multi_band_list)):
            if compute_bool[i] is True:
                kwargs_data = multi_band_list[i][0]
                kwargs_psf = multi_band_list[i][1]
                kwargs_numerics = multi_band_list[i][2]
                index_lens_list = multi_band_list[i][3].get('index_lens_list', [k for k in range(len(lens_model_list))])
                self._idex_lens_list.append(index_lens_list)
                lens_model_list_sub = [lens_model_list[k] for k in index_lens_list]
                lens_model_class = LensModel(lens_model_list=lens_model_list_sub)
                data_i = Data(kwargs_data=kwargs_data)
                psf_i = PSF(kwargs_psf=kwargs_psf)
                point_source_class_i = copy.deepcopy(point_source_class)
                imageModel = ImageModel(data_i, psf_i, lens_model_class, source_model_class,
                                        lens_light_model_class, point_source_class_i,
                                        kwargs_numerics=kwargs_numerics)
                imageModel_list.append(imageModel)
        super(MultiFrame, self).__init__(imageModel_list)

    def image_linear_solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, inv_bool=False):
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
        A = self.linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        C_D_response, model_error_list = self.error_response(kwargs_lens, kwargs_ps)
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        kwargs_lens_0 = [kwargs_lens[k] for k in self._idex_lens_list[0]]
        _, _, _, _ = self._imageModel_list[0]._update_linear_kwargs(param, kwargs_lens_0, kwargs_source, kwargs_lens_light, kwargs_ps)
        wls_list = self._array2image_list(wls_model)
        return wls_list, model_error_list, cov_param, param

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        A = []
        for i in range(self._num_bands):
            kwargs_lens_i = [kwargs_lens[k] for k in self._idex_lens_list[i]]
            A_i = self._imageModel_list[i].linear_response_matrix(kwargs_lens_i, kwargs_source, kwargs_lens_light, kwargs_ps)
            if i == 0:
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
            d_i = self._imageModel_list[i].data_response
            if i == 0:
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
            num_data = self.num_response_list[i]
            array_i = array[k:k + num_data]
            image_i = self._imageModel_list[i].ImageNumerics.array2image(array_i)
            image_list.append(image_i)
            k += num_data
        return image_list

    def error_response(self, kwargs_lens, kwargs_ps):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        C_D_response, model_error = [], []
        for i in range(self._num_bands):
            kwargs_lens_i = [kwargs_lens[k] for k in self._idex_lens_list[i]]
            C_D_response_i, model_error_i = self._imageModel_list[i].error_response(kwargs_lens_i, kwargs_ps)
            model_error.append(model_error_i)
            if i == 0:
                C_D_response = C_D_response_i
            else:
                C_D_response = np.append(C_D_response, C_D_response_i)
        return C_D_response, model_error

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg=False,
                                    compute_bool=None):
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
        im_sim_list, model_error_list, cov_matrix, param = self.image_linear_solve(kwargs_lens, kwargs_source,
                                                                         kwargs_lens_light, kwargs_ps,
                                                                         inv_bool=source_marg)
        # compute X^2
        logL = 0
        for i in range(self._num_bands):
            logL += self._imageModel_list[i].Data.log_likelihood(im_sim_list[i], self._imageModel_list[i].ImageNumerics.mask, model_error_list[i])
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalisation_const(cov_matrix)
            logL += marg_const
        return logL

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, compute_bool=None):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        #TODO this routine might not rightfully compute the number of point sources that are present in all frames when solving for their positions
        return self._imageModel_list[0].num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
