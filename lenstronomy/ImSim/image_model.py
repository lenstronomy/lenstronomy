__author__ = 'sibirrer'

from lenstronomy.ImSim.image_numerics import ImageNumerics
import lenstronomy.ImSim.de_lens as de_lens


import numpy as np


class ImageModel(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None, lens_light_model_class=None, point_source_class=None, kwargs_numerics={}):
        """


        :param kwargs_options: keywords of the modelling choices
        'subgrid_res': integer, sub-grid ray-tracing resolution
        'psf_subgrid': bool, if True, performs the convolution on the subgrid resolution
        (higher accuracy for higher computational cost)
        :param kwargs_data: keywords of the data, see Data() class for further information
        :param kwargs_psf: keywords of the PSF convolution, see PSF() class for further information
        """
        self.PSF = psf_class
        self.Data = data_class
        self.kwargs_numerics = kwargs_numerics
        self.ImageNumerics = ImageNumerics(data=self.Data, psf=self.PSF, kwargs_numerics=kwargs_numerics)
        self.LensModel = lens_model_class
        self.PointSource = point_source_class
        if self.PointSource is not None:
            self.PointSource.update_lens_model(lens_model_class=lens_model_class)
            if self.PSF.psf_error_map is not None:
                self._psf_error_map = kwargs_numerics.get('psf_error_map', True)
            else:
                self._psf_error_map = False
        else:
            self._psf_error_map = False
        self.SourceModel = source_model_class
        self.LensLightModel = lens_light_model_class

    def update_data(self, data_class):
        """

        :param data_class: instance of Data() class
        :return: no return. Class is updated.
        """
        self.Data = data_class
        self.ImageNumerics._Data = data_class

    def update_psf(self, psf_class):
        """

        update the instance of the class with a new instance of PSF() with a potentially different point spread function

        :param psf_class:
        :return: no return. Class is updated.
        """
        self.PSF = psf_class
        self.ImageNumerics._PSF = psf_class

    def update_numerics(self, kwargs_numerics):
        """

        update numerical options

        :param kwargs_numerics:
        :return: no return. Class is updated.
        """
        self._psf_error_map = kwargs_numerics.get('psf_error_map', False)
        self.ImageNumerics = ImageNumerics(data=self.Data, psf=self.PSF, kwargs_numerics=kwargs_numerics)

    def source_surface_brightness(self, kwargs_source, kwargs_lens=None, unconvolved=False, de_lensed=False, k=None):
        """

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: 1d array of surface brightness pixels
        """
        if self.SourceModel is None:
            return np.zeros_like(self.Data.data)
        if de_lensed is True or self.LensModel is None:
            x_source, y_source = self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting
        else:
            x_source, y_source = self.LensModel.ray_shooting(self.ImageNumerics.ra_grid_ray_shooting,
                                                             self.ImageNumerics.dec_grid_ray_shooting, kwargs_lens)
        source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source, k=k)
        source_light_final = self.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """

        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: 1d array of surface brightness pixels
        """
        if self.LensLightModel is None:
            return np.zeros_like(self.Data.data)
        lens_light = self.LensLightModel.surface_brightness(self.ImageNumerics.ra_grid_ray_shooting,
                                                            self.ImageNumerics.dec_grid_ray_shooting,
                                                            kwargs_lens_light, k=k)
        lens_light_final = self.ImageNumerics.re_size_convolve(lens_light, unconvolved=unconvolved)
        return lens_light_final

    def point_source(self, kwargs_ps, kwargs_lens=None, unconvolved=False, k=None):
        """

        computes the point source positions and paints PSF convolutions on them

        :param kwargs_ps:
        :param k:
        :return:
        """
        point_source_image = np.zeros_like(self.Data.data)
        if unconvolved or self.PointSource is None:
            return point_source_image
        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=True, k=k)
        for i in range(n_points):
            point_source_image += self.ImageNumerics.point_source_rendering(ra_pos[i], dec_pos[i], amp[i])
        return point_source_image

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, inv_bool=False):
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
        if not self.LensModel is None:
            x_source, y_source = self.LensModel.ray_shooting(self.ImageNumerics.ra_grid_ray_shooting,
                                                         self.ImageNumerics.dec_grid_ray_shooting, kwargs_lens)
        else:
            x_source, y_source = self.ImageNumerics.ra_grid_ray_shooting, self.ImageNumerics.dec_grid_ray_shooting

        A = self._response_matrix(self.ImageNumerics.ra_grid_ray_shooting,
                                             self.ImageNumerics.dec_grid_ray_shooting, x_source, y_source,
                                             kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, self.ImageNumerics.mask)
        error_map = self.error_map(kwargs_lens, kwargs_ps)
        error_map = self.ImageNumerics.image2array(error_map)
        d = self.ImageNumerics.image2array(self.Data.data*self.ImageNumerics.mask)
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / (self.ImageNumerics.C_D_response + error_map), d, inv_bool=inv_bool)
        _, _, _, _ = self._update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        model = self.ImageNumerics.array2image(wls_model)
        error_map = self.ImageNumerics.array2image(error_map)
        return model, error_map, cov_param, param

    def image(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        """

        make a image with a realisation of linear parameter values "param"

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 1d array of surface brightness pixels of the simulation
        """
        if source_add:
            source_light = self.source_surface_brightness(kwargs_source, kwargs_lens, unconvolved=unconvolved)
        else:
            source_light = np.zeros_like(self.Data.data)
        if lens_light_add:
            lens_light = self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        else:
            lens_light = np.zeros_like(self.Data.data)
        if point_source_add:
            point_source = self.point_source(kwargs_ps, kwargs_lens, unconvolved=unconvolved)
        else:
            point_source = np.zeros_like(self.Data.data)
        model = (source_light + lens_light + point_source) * self.ImageNumerics.mask
        return model

    def error_map(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        error_map = np.zeros_like(self.Data.data)
        if self._psf_error_map:
            ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens)
            for i in range(0, n_points):
                error_map_add = self.ImageNumerics.psf_error_map(ra_pos[i], dec_pos[i], amp[i])
                error_map += error_map_add
        return error_map

    def point_sources_list(self, kwargs_ps, kwargs_lens):
        """

        :param kwargs_ps:
        :return: list of images containing only single point sources
        """
        point_list = []
        if self.PointSource is None:
            return point_list
        ra_array, dec_array, amp_array = self.PointSource.point_source_list(kwargs_ps, kwargs_lens)
        for i in range(len(ra_array)):
            point_source = self.ImageNumerics.point_source_rendering([ra_array[i]], [dec_array[i]], [amp_array[i]])
            point_list.append(point_source)
        return point_list

    def image_positions(self, kwargs_ps, kwargs_lens):
        """
        lens equation solver for image positions given lens model and source position (only for image positions within
        the data frame).

        :param kwargs_lens: keyword arguments of lens models (as list)
        :param sourcePos_x: source position in relative arc sec
        :param sourcePos_y: source position in relative arc sec
        :return: x_coords, y_coords of image positions
        """
        if self.PointSource is None:
            return [], []
        x_mins, y_mins = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        return x_mins, y_mins

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, source_marg=False):
        """

        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: log likelihood (natural logarithm)
        """
        # generate image
        im_sim, model_error, cov_matrix, param = self.image_linear_solve(kwargs_lens, kwargs_source,
                                                                                   kwargs_lens_light, kwargs_else,
                                                                                   inv_bool=source_marg)
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.ImageNumerics.mask, model_error)
        if cov_matrix is not None and source_marg:
            marg_const = de_lens.marginalisation_const(cov_matrix)
            #if marg_const + logL > 0:
            #logL = np.log(np.exp(logL) + np.exp(marg_const))
            logL += marg_const
        return logL

    def reduced_residuals(self, model, error_map=0):
        """

        :param model:
        :return:
        """
        mask = self.ImageNumerics.mask
        residual = (model - self.Data.data)/np.sqrt(self.Data.C_D+np.abs(error_map))*mask
        return residual

    def reduced_chi2(self, model, error_map=0):
        """
        returns reduced chi2
        :param model:
        :param error_map:
        :return:
        """
        chi2 = self.reduced_residuals(model, error_map)
        return np.sum(chi2**2) / self.numData_evaluate

    @property
    def numData_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return self.ImageNumerics.numData_evaluate

    def fermat_potential(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        ra_pos_list, dec_pos_list = self.PointSource.image_position(kwargs_ps, kwargs_lens)
        ra_source_list, dec_source_list = self.PointSource.source_position(kwargs_ps, kwargs_lens)
        phi_fermat = []
        for i in range(len(ra_pos_list)):
            phi_fermat_i = self.LensModel.fermat_potential(ra_pos_list[i], dec_pos_list[i], ra_source_list[i],
                                                           dec_source_list[i], kwargs_lens)
            phi_fermat.append(phi_fermat_i)
        return phi_fermat

    def _response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, mask, unconvolved=False):
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
            source_light_response, n_source = self.SourceModel.functions_split(x_source, y_source, kwargs_source)
        else:
            source_light_response, n_source = [], 0
        if not self.LensLightModel is None:
            lens_light_response, n_lens_light = self.LensLightModel.functions_split(x_grid, y_grid,                                                                               kwargs_lens_light)
        else:
            lens_light_response, n_lens_light = [], 0
        if not self.PointSource is None:
            ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(kwargs_ps, kwargs_lens, with_amp=False)
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
        # error_map
        #error_map = np.zeros(num_response)
        #if self._psf_error_map:
        #    for i in range(0, n_points):
        #        error_map_add = self.ImageNumerics.psf_error_map(ra_pos[i], dec_pos[i], amp[i])
        #        error_map += self.ImageNumerics.image2array(error_map_add)
        return A#, error_map

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
        if not self.SourceModel is None:
            kwargs_source, i = self.SourceModel.update_linear(param, i, kwargs_list=kwargs_source)
        if not self.LensLightModel is None:
            kwargs_lens_light, i = self.LensLightModel.update_linear(param, i, kwargs_list=kwargs_lens_light)
        if not self.PointSource is None:
            kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
