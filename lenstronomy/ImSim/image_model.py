__author__ = 'sibirrer'

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LensLightModel, SourceModel
from lenstronomy.ImSim.point_source import PointSource
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.Data.imaging_data import Data


import numpy as np


class ImageModel(object):
    """
    this class uses functions of lens_model and source_model to make a lensed image
    """
    def __init__(self, kwargs_options, kwargs_data, kwargs_psf=None):
        self.Data = Data(kwargs_data, subgrid_res=kwargs_options.get('subgrid_res', 1),
                         psf_subgrid=kwargs_options.get('psf_subgrid', False))
        self.LensModel = LensModel(lens_model_list=kwargs_options['lens_model_list'])
        self.SourceModel = SourceModel(kwargs_options.get('source_light_model_list', ['NONE']))
        self.LensLightModel = LensLightModel(kwargs_options.get('lens_light_model_list', ['NONE']))
        self.PointSource = PointSource(self.Data, point_source=kwargs_options.get('point_source', False),
                                       fix_magnification=kwargs_options.get('fix_magnification', False),
                                       error_map=kwargs_options.get('error_map', False),
                                       fix_error_map=kwargs_options.get('fix_error_map', False))
        self._kwargs_options = kwargs_options
        self._kwargs_psf = kwargs_psf
        self.imagePosition = LensEquationSolver(lens_model_list=kwargs_options['lens_model_list'])

    def source_surface_brightness(self, kwargs_lens, kwargs_source, kwargs_else, unconvolved=False, de_lensed=False):
        """
        computes the source surface brightness distribution

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: 1d array of surface brightness pixels
        """

        if de_lensed is True:
            x_source, y_source = self.Data.x_grid_sub, self.Data.y_grid_sub
        else:
            x_source, y_source = self.LensModel.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens)
        source_light = self.SourceModel.surface_brightness(x_source, y_source, kwargs_source)
        source_light_final = self.Data.re_size_convolve(source_light, self._kwargs_psf, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False):
        """
        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: 1d array of surface brightness pixels
        """
        lens_light = self.LensLightModel.surface_brightness(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens_light)
        lens_light_final = self.Data.re_size_convolve(lens_light, self._kwargs_psf, unconvolved=unconvolved)
        return lens_light_final

    def image_linear_solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=False):
        """
        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        map_error = self._kwargs_options.get('error_map', False)
        x_source, y_source = self.LensModel.ray_shooting(self.Data.x_grid_sub, self.Data.y_grid_sub, kwargs_lens)
        A, error_map = self._response_matrix(self.Data.x_grid_sub, self.Data.y_grid_sub, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, self.Data.mask, map_error=map_error)
        d = self.Data.image2array(self.Data.data*self.Data.mask)
        param, cov_param, wls_model = de_lens.get_param_WLS(A.T, 1 / (self.Data.C_D_response + error_map), d, inv_bool=inv_bool)
        _, _, _, _ = self._update_linear_kwargs(param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        model = self.Data.array2image(wls_model)
        error_map = self.Data.array2image(error_map)
        return model, error_map, cov_param, param

    def image_with_params(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        """
        make a image with a realisation of linear parameter values "param"

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 1d array of surface brightness pixels of the simulation
        """
        if source_add:
            source_light = self.source_surface_brightness(kwargs_lens, kwargs_source, kwargs_else, unconvolved=unconvolved)
        else:
            source_light = np.zeros_like(self.Data.data)
        if lens_light_add:
            lens_light = self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        else:
            lens_light = np.zeros_like(self.Data.data)
        if point_source_add and self._kwargs_options.get('point_source', False):
            point_source, error_map = self.PointSource.point_source(self._kwargs_psf, kwargs_else)
        else:
            point_source = np.zeros_like(self.Data.data)
            error_map = np.zeros_like(self.Data.data)
        model = (source_light + lens_light + point_source) * self.Data.mask
        return model, error_map

    def point_sources_list(self, kwargs_else):
        """

        :param kwargs_else:
        :return: list of images containing only single point sources
        """
        return self.PointSource.point_source_list(self._kwargs_psf, kwargs_else)

    def image_positions(self, kwargs_lens, sourcePos_x, sourcePos_y):
        """
        lens equation solver for image positions given lens model and source position

        :param kwargs_lens: keyword arguments of lens models (as list)
        :param sourcePos_x: source position in relative arc sec
        :param sourcePos_y: source position in relative arc sec
        :return: x_coords, y_coords of image positions
        """
        deltaPix = self.Data.deltaPix / 2.
        numPix = self.Data.nx * 2
        x_mins, y_mins = self.imagePosition.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=deltaPix, search_window=deltaPix*numPix)
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
        logL = self.Data.log_likelihood(im_sim, model_error)
        # logL = self.compare.get_log_likelihood(X, cov_matrix=cov_matrix)
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
        mask = self.Data.mask
        residual = (model - self.Data.data)/np.sqrt(self.Data.C_D+np.abs(error_map))*mask
        return residual

    @property
    def numData_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return self.Data.numData_evaluate

    def fermat_potential(self, kwargs_lens, kwargs_else):
        """

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        if 'ra_pos' in kwargs_else and 'dec_pos' in kwargs_else:
            ra_pos = kwargs_else['ra_pos']
            dec_pos = kwargs_else['dec_pos']
        else:
            raise ValueError('No point source positions assigned')
        ra_source, dec_source = self.LensModel.ray_shooting(ra_pos, dec_pos, kwargs_lens)
        ra_source = np.mean(ra_source)
        dec_source = np.mean(dec_source)
        phi_fermat = self.LensModel.fermat_potential(ra_pos, dec_pos, ra_source, dec_source, kwargs_lens)
        return phi_fermat

    def _response_matrix(self, x_grid, y_grid, x_source, y_source, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, mask, map_error=False, unconvolved=False):
        """
        return linear response Matrix
        :param x_grid:
        :param y_grid:
        :param x_source:
        :param y_source:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :param mask:
        :param map_error:
        :param unconvolved:
        :return:
        """
        kwargs_psf = self._kwargs_psf
        source_light_response, n_source = self.SourceModel.lightModel.functions_split(x_source, y_source, kwargs_source)
        lens_light_response, n_lens_light = self.LensLightModel.lightModel.functions_split(x_grid, y_grid,
                                                                                           kwargs_lens_light)
        n_points = self.PointSource.num_basis(kwargs_else)
        num_param = n_points + n_lens_light + n_source

        num_response = self.Data.num_response
        A = np.zeros((num_param, num_response))
        error_map = np.zeros(num_response)
        n = 0
        # response of sersic source profile
        for i in range(0, n_source):
            image = source_light_response[i]
            image = self.Data.re_size_convolve(image, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = self.Data.image2array(image)
            n += 1
        # response of lens light profile
        for i in range(0, n_lens_light):
            image = lens_light_response[i]
            image = self.Data.re_size_convolve(image, kwargs_psf, unconvolved=unconvolved)
            A[n, :] = self.Data.image2array(image)
            n += 1
        # response of point sources
        if self._kwargs_options.get('point_source', False):
            if self._kwargs_options.get('fix_magnification', False):
                mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens)
            else:
                mag = np.ones_like(kwargs_else['ra_pos'])
            A_point, error_map = self.PointSource.point_source_response(kwargs_psf, kwargs_else, point_amp=mag, map_error=map_error)
            A[n:n+n_points, :] = A_point
            n += n_points
        A = self._add_mask(A, mask)
        return A, error_map

    def _add_mask(self, A, mask):
        """

        :param A: 2d matrix n*len(mask)
        :param mask: 1d vector of 1 or zeros
        :return: column wise multiplication of A*mask
        """
        return A[:] * self.Data.image2array(mask)

    def _update_linear_kwargs(self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
        """
        links linear parameters to kwargs arguments

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        for k, model in enumerate(self._kwargs_options['source_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_source[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_source[k]['I0_2'] = param[i]
                i += 1
            if model in ['BULDGE_DISK']:
                kwargs_source[k]['I0_b'] = param[i]
                i += 1
                kwargs_source[k]['I0_d'] = param[i]
                i += 1
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_source[k]['sigma0'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_source[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_source[k]['amp'] = param[i:i+num_param]
                i += num_param
            if model in ['UNIFORM']:
                kwargs_source[k]['mean'] = param[i]
                i += 1
        for k, model in enumerate(self._kwargs_options['lens_light_model_list']):
            if model in ['SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC', 'CORE_SERSIC']:
                kwargs_lens_light[k]['I0_sersic'] = param[i]
                i += 1
            if model in ['DOUBLE_SERSIC', 'DOUBLE_CORE_SERSIC']:
                kwargs_lens_light[k]['I0_2'] = param[i]
                i += 1
            if model in ['BULDGE_DISK']:
                kwargs_lens_light[k]['I0_b'] = param[i]
                i += 1
                kwargs_lens_light[k]['I0_d'] = param[i]
                i += 1
            if model in ['HERNQUIST', 'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE']:
                kwargs_lens_light[k]['sigma0'] = param[i]
                i += 1
            if model in ['SHAPELETS']:
                n_max = kwargs_lens_light[k]['n_max']
                num_param = (n_max + 1) * (n_max + 2) / 2
                kwargs_lens_light[k]['amp'] = param[i:i+num_param]
                i += num_param
            if model in ['UNIFORM']:
                kwargs_lens_light[k]['mean'] = param[i]
                i += 1
        num_images = self._kwargs_options.get('num_images', 0)
        if num_images > 0 and self._kwargs_options['point_source']:
            if self._kwargs_options.get('fix_magnification', False):
                mag = self.LensModel.magnification(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens)
                kwargs_else['point_amp'] = np.abs(mag) * param[i]
                i += 1
            else:
                n_points = len(kwargs_else['ra_pos'])
                kwargs_else['point_amp'] = param[i:i+n_points]
                i += n_points
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else
