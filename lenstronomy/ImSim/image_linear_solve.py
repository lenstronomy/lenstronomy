from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
import numpy as np


class ImageModelLinear(ImageModel):
    """
    linear version class, inherits ImageModel
    """

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           inv_bool=False):
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
        C_D_response, model_error = self.error_response(kwargs_lens, kwargs_ps)
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _, _, _ = self._update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.ImageNumerics.array2image(wls_model)
        return model, model_error, cov_param, param

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        A = self._response_matrix(self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting,
                                  kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, self.ImageNumerics.mask)
        return A

    @property
    def data_response(self):
        """
        returns the 1d array of the data element that is fitted for (including masking)

        :return: 1d numpy array
        """
        d = self.ImageNumerics.image2array(self.Data.data * self.ImageNumerics.mask)
        return d

    def error_response(self, kwargs_lens, kwargs_ps):
        """
        returns the 1d array of the error estimate corresponding to the data response

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        model_error = self.error_map(kwargs_lens, kwargs_ps)
        C_D_response = self.ImageNumerics.image2array(self.Data.C_D + model_error)
        return C_D_response, model_error

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg=False):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: log likelihood (natural logarithm)
        """
        # generate image
        im_sim, model_error, cov_matrix, param = self.image_linear_solve(kwargs_lens, kwargs_source,
                                                                         kwargs_lens_light, kwargs_ps,
                                                                         inv_bool=source_marg)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.ImageNumerics.mask, model_error)
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalisation_const(cov_matrix)
            # if marg_const + logL > 0:
            # logL = np.log(np.exp(logL) + np.exp(marg_const))
            logL += marg_const
        return logL

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        num += self.SourceModel.num_param_linear(kwargs_source)
        num += self.LensLightModel.num_param_linear(kwargs_lens_light)
        num += self.PointSource.num_basis(kwargs_ps, kwargs_lens)
        return num

    def _response_matrix(self, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, mask,
                         unconvolved=False):
        """

        return linear response Matrix

        :param x_grid:
        :param y_grid:
        :param x_source:
        :param y_source:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param mask:
        :param unconvolved:
        :return:
        """
        if not self.SourceModel is None:
            source_light_response, n_source = self.source_mapping.image_flux_split(x_grid, y_grid, kwargs_lens,
                                                                                   kwargs_source)
        else:
            source_light_response, n_source = [], 0
        if not self.LensLightModel is None:
            lens_light_response, n_lens_light = self.LensLightModel.functions_split(x_grid, y_grid, kwargs_lens_light)
        else:
            lens_light_response, n_lens_light = [], 0
        if not self.PointSource is None:
            ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens,
                                                                                  with_amp=False)
        else:
            ra_pos, dec_pos, amp, n_points = [], [], [], 0
        num_param = n_points + n_lens_light + n_source

        num_response = self.ImageNumerics.num_response
        A = np.zeros((num_param, num_response))
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        # response of point sources
        for i in range(0, n_points):
            image = self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
            A[n, :] = self.ImageNumerics.image2array(image)
            n += 1
        A = self._add_mask(A, mask)
        return np.nan_to_num(A)

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * self.ImageNumerics.image2array(mask)

    def _update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        if self.SourceModel is not None:
            kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        if self.LensLightModel is not None:
            kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        if self.PointSource is not None:
            kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
