from lenstronomy.ImSim.image_solve import ImageFit
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.Util import util
import numpy as np


class ImageLinearFit(ImageFit):
    """
    linear inversion class, inherits ImageFit
    """
    def __init__(self, data_class, psf_class, lens_model_class=None, source_model_class=None,
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
        super(ImageLinearFit, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                             source_model_class=source_model_class,
                                             lens_light_model_class=lens_light_model_class,
                                             point_source_class=point_source_class, extinction_class=extinction_class, kwargs_numerics=kwargs_numerics)

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
        return self._image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction,
                                        kwargs_special, inv_bool=inv_bool)

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
        d = self.data_response
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / C_D_response, d, inv_bool=inv_bool)
        _, _, _, _ = self.update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.array_masked2image(wls_model)
        return model, model_error, cov_param, param

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
        A = self._linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction, kwargs_special)
        return A

    def likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                    kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: log likelihood (natural logarithm)
        """
        return self._likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                 kwargs_extinction, kwargs_special, source_marg, linear_prior=linear_prior)

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
        im_sim, model_error, cov_matrix, param = self._image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light,
                                                                          kwargs_ps, kwargs_extinction, kwargs_special,
                                                                          inv_bool=source_marg)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalization_new(cov_matrix, d_prior=linear_prior)
            logL += marg_const
        return logL

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        return self._num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    def _num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        num += self.SourceModel.num_param_linear(kwargs_source)
        num += self.LensLightModel.num_param_linear(kwargs_lens_light)
        num += self.PointSource.num_basis(kwargs_ps, kwargs_lens)
        return num

    def _linear_response_matrix(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                kwargs_extinction=None, kwargs_special=None, unconvolved=False):
        """

        return linear response Matrix

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param unconvolved:
        :return:
        """
        x_grid, y_grid = self.ImageNumerics.coordinates_evaluate
        source_light_response, n_source = self.source_mapping.image_flux_split(x_grid, y_grid, kwargs_lens,
                                                                               kwargs_source)
        extinction = self._extinction.extinction(x_grid, y_grid, kwargs_extinction=kwargs_extinction,
                                                 kwargs_special=kwargs_special)
        lens_light_response, n_lens_light = self.LensLightModel.functions_split(x_grid, y_grid, kwargs_lens_light)

        ra_pos, dec_pos, amp, n_points = self.point_source_linear_response_set(kwargs_ps, kwargs_lens, kwargs_special, with_amp=False)
        num_param = n_points + n_lens_light + n_source

        num_response = self.num_data_evaluate
        A = np.zeros((num_param, num_response))
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image *= extinction
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.image2array_masked(image)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = self.image2array_masked(image)
            n += 1
        # response of point sources
        for i in range(0, n_points):
            image = self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
            A[n, :] = self.image2array_masked(image)
            n += 1
        return np.nan_to_num(A)

    def _source_linear_response_matrix(self, x_grid, y_grid, kwargs_lens, kwargs_source):
        """

        return linear response Matrix for source light only

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param unconvolved:
        :return:
        """
        source_light_response, n_source = self.source_mapping.image_flux_split(x_grid, y_grid, kwargs_lens,
                                                                               kwargs_source)
        return source_light_response, n_source

    def update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def point_source_linear_response_set(self, kwargs_ps, kwargs_lens, kwargs_special, with_amp=True):
        """

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param kwargs_special: special keyword argument list, may include 'delta_x_image' and 'delta_y_image'
        :param with_amp: bool, if True, relative magnification between multiply imaged point sources are held fixed.
        :return: list of positions and amplitudes split in different basis components with applied astrometric corrections
        """

        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=with_amp)

        if kwargs_special is not None:
            if 'delta_x_image' in kwargs_special:
                delta_x, delta_y = kwargs_special['delta_x_image'], kwargs_special['delta_y_image']
                k = 0
                n = len(delta_x)
                for i in range(n_points):
                    for j in range(len(ra_pos[i])):
                        if k >= n:
                            break
                        ra_pos[i][j] = ra_pos[i][j] + delta_x[k]
                        dec_pos[i][j] = dec_pos[i][j] + delta_y[k]
                        k += 1
        return ra_pos, dec_pos, amp, n_points
