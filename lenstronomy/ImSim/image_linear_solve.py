from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
from lenstronomy.Util import util
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
import numpy as np

__all__ = ["ImageLinearFit"]


class ImageLinearFit(ImageModel):
    """Linear version class, inherits ImageModel.

    When light models use pixel-based profile types, such as 'SLIT_STARLETS', the WLS
    linear inversion is replaced by the regularized inversion performed by an external
    solver. The current pixel-based solver is provided by the SLITronomy plug-in.
    """

    def __init__(
        self,
        data_class,
        psf_class=None,
        lens_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_class=None,
        extinction_class=None,
        kwargs_numerics=None,
        likelihood_mask=None,
        psf_error_map_bool_list=None,
        kwargs_pixelbased=None,
        linear_solver=True,
    ):
        """:param data_class: ImageData() instance :param psf_class: PSF() instance
        :param lens_model_class: LensModel() instance :param source_model_class:
        LightModel() instance :param lens_light_model_class: LightModel() instance
        :param point_source_class: PointSource() instance :param kwargs_numerics:
        keyword arguments passed to the Numerics module :param likelihood_mask: 2d
        boolean array of pixels to be counted in the likelihood calculation/linear
        optimization :param psf_error_map_bool_list: list of boolean of length of point
        source models.

        Indicates whether PSF error map is used for the point source model stated as the
        index.
        :param kwargs_pixelbased: keyword arguments with various settings related to the
                pixel-based solver          (see SLITronomy documentation) being applied
                to the point sources.
        :param linear_solver: bool, if True (default) fixes the linear amplitude
                parameters 'amp' (avoid sampling) such          that they get
                overwritten by the linear solver solution.
        """
        super(ImageLinearFit, self).__init__(
            data_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_class=point_source_class,
            extinction_class=extinction_class,
            kwargs_numerics=kwargs_numerics,
            kwargs_pixelbased=kwargs_pixelbased,
        )
        self._linear_solver = linear_solver
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(
                self.PointSource.point_source_type_list
            )
        self._psf_error_map_bool_list = psf_error_map_bool_list
        if likelihood_mask is None:
            likelihood_mask = np.ones_like(data_class.data)
        self.likelihood_mask = np.array(likelihood_mask, dtype=bool)
        self._mask1d = util.image2array(self.likelihood_mask)
        if self._pixelbased_bool is True:
            # update the pixel-based solver with the likelihood mask
            self.PixelSolver.set_likelihood_mask(self.likelihood_mask)

        # prepare to use fft convolution for the natwt linear solver
        if self.Data.likelihood_method() == "interferometry_natwt":
            self._convolution = PixelKernelConvolution(
                kernel=self.PSF.kernel_point_source
            )

    def image_linear_solve(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        inv_bool=False,
    ):
        """Computes the image (lens and source surface brightness with a given lens
        model). The linear parameters are computed with a weighted linear least square
        optimization (i.e. flux normalization of the brightness profiles) However in
        case of pixel-based modelling, pixel values are constrained by an external
        solver (e.g. SLITronomy).

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
                of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
                lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
                external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the
                purpose of the covariance matrix.
        :return: 2d array of surface brightness pixels of the optimal solution of the
                linear parameters to match the data
        """
        return self._image_linear_solve(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
            inv_bool=inv_bool,
        )

    def _image_linear_solve(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        inv_bool=False,
    ):
        """Computes the image (lens and source surface brightness with a given lens
        model). By default, the linear parameters are computed with a weighted linear
        least square optimization (i.e. flux normalization of the brightness profiles)
        However in case of pixel-based modelling, pixel values are constrained by an
        external solver (e.g. SLITronomy).

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
                of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
                lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
                external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the
                purpose of the covariance matrix. This has no impact in case of pixel-
                based             modelling.
        :return: 2d array of surface brightness pixels of the optimal solution of the
                linear parameters to match the data
        """
        if self._pixelbased_bool is True:
            model, model_error, cov_param, param = self.image_pixelbased_solve(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
            )
        elif self.Data.likelihood_method() == "diagonal":
            A = self._linear_response_matrix(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
            )
            C_D_response, model_error = self._error_response(
                kwargs_lens, kwargs_ps, kwargs_special=kwargs_special
            )
            d = self.data_response
            param, cov_param, wls_model = de_lens.get_param_WLS(
                A.T, 1 / C_D_response, d, inv_bool=inv_bool
            )
            model = self.array_masked2image(wls_model)
            _, _, _, _ = self._update_linear_kwargs(
                param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
            )
        elif self.Data.likelihood_method() == "interferometry_natwt":
            (
                model,
                model_error,
                cov_param,
                param,
            ) = self._image_linear_solve_interferometry_natwt(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
            )
        else:
            raise ValueError(
                "likelihood_method %s not supported!" % self.Data.likelihood_method()
            )
        return model, model_error, cov_param, param

    def image_pixelbased_solve(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        init_lens_light_model=None,
    ):
        """Computes the image (lens and source surface brightness with a given lens
        model) using the pixel-based solver.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
                of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
                lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to point sources
        :param kwargs_extinction: keyword arguments corresponding to dust extinction
        :param kwargs_special: keyword arguments corresponding to "special" parameters
        :param init_lens_light_model: optional initial guess for the lens surface
            brightness
        :return: 2d array of surface brightness pixels of the optimal solution of the
                linear parameters to match the data
        """
        _, model_error = self._error_response(
            kwargs_lens, kwargs_ps, kwargs_special=kwargs_special
        )
        model, param, _ = self.PixelSolver.solve(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=kwargs_ps,
            kwargs_special=kwargs_special,
            init_lens_light_model=init_lens_light_model,
        )
        cov_param = None
        _, _ = self.update_pixel_kwargs(kwargs_source, kwargs_lens_light)
        _, _, _, _ = self._update_linear_kwargs(
            param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )
        return model, model_error, cov_param, param

    def linear_response_matrix(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
    ):
        """Computes the linear response matrix (m x n), with n being the data size and m
        being the coefficients.

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_source: extended source model keyword argument list
        :param kwargs_lens_light: lens light model keyword argument list
        :param kwargs_ps: point source model keyword argument list
        :param kwargs_extinction: extinction model keyword argument list
        :param kwargs_special: special keyword argument list
        :return: linear response matrix
        """
        A = self._linear_response_matrix(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
        )
        return A

    @property
    def data_response(self):
        """Returns the 1d array of the data element that is fitted for (including
        masking)

        :return: 1d numpy array
        """
        d = self.image2array_masked(self.Data.data)
        return d

    def error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """Returns the 1d array of the error estimate corresponding to the data
        response.

        :return: 1d numpy array of response, 2d array of additional errors (e.g. point
                source uncertainties)
        """
        return self._error_response(
            kwargs_lens, kwargs_ps, kwargs_special=kwargs_special
        )

    def _error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """Returns the 1d array of the error estimate corresponding to the data
        response.

        :return: 1d numpy array of response, 2d array of additional errors (e.g. point
                source uncertainties)
        """
        model_error = self._error_map_model(
            kwargs_lens, kwargs_ps, kwargs_special=kwargs_special
        )
        # adding the uncertainties estimated from the data with the ones from the model
        C_D_response = self.image2array_masked(self.Data.C_D + model_error)
        return C_D_response, model_error

    def likelihood_data_given_model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        linear_solver=True,
    ):
        """Computes the likelihood of the data given a model This is specified with the
        non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
                of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
                lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
                external shear and point source image positions
        :param kwargs_extinction: 
        :param kwargs_special: 
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :param check_positive_flux: bool, if True, checks whether the linear inversion
                resulted in non-negative flux components and applies a punishment in the
                likelihood if so.
        :param linear_solver: bool, if True (default) fixes the linear amplitude
                parameters 'amp' (avoid sampling) such that they get overwritten by the
                linear solver solution.
        :return: log likelihood (natural logarithm)
        """
        return self._likelihood_data_given_model(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
            source_marg,
            linear_prior=linear_prior,
            check_positive_flux=check_positive_flux,
            linear_solver=linear_solver,
        )

    def _likelihood_data_given_model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        linear_solver=True,
    ):
        """Computes the likelihood of the data given a model This is specified with the
        non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
                of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
                lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
                external shear and point source image positions
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :param check_positive_flux: bool, if True, checks whether the linear inversion
                resulted in non-negative flux components and applies a punishment in the
                likelihood if so.
        :param linear_solver: bool, if True (default) fixes the linear amplitude
                parameters 'amp' (avoid sampling) such that they get overwritten by the
                linear solver solution.
        :return: log likelihood (natural logarithm)
        """
        # generate image
        if linear_solver is False:
            im_sim = self.image(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
            )
            cov_matrix = None
            model_error = self._error_map_model(
                kwargs_lens, kwargs_ps=kwargs_ps, kwargs_special=kwargs_special
            )
        else:
            im_sim, model_error, cov_matrix, param = self._image_linear_solve(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
                inv_bool=source_marg,
            )
        # compute X^2
        logL = self.Data.log_likelihood(im_sim, self.likelihood_mask, model_error)

        if self._pixelbased_bool is False:
            if cov_matrix is not None and source_marg:
                marg_const = de_lens.marginalization_new(
                    cov_matrix, d_prior=linear_prior
                )
                logL += marg_const
        if check_positive_flux is True:
            bool_ = self.check_positive_flux(
                kwargs_source, kwargs_lens_light, kwargs_ps
            )
            if bool_ is False:
                logL -= 10**8
        return logL

    def num_param_linear(
        self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    ):
        """:return: number of linear coefficients to be solved for in the linear
        inversion."""
        return self._num_param_linear(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )

    def _num_param_linear(
        self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    ):
        """:return: number of linear coefficients to be solved for in the linear
        inversion."""
        num = 0
        if self._pixelbased_bool is False:
            num += self.SourceModel.num_param_linear(kwargs_source)
            num += self.LensLightModel.num_param_linear(kwargs_lens_light)
        num += self.PointSource.num_basis(kwargs_ps, kwargs_lens)
        return num

    def _linear_response_matrix(
        self,
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
    ):
        """Computes the linear response matrix (m x n), with n being the data size and m
        being the coefficients.

        The calculation is done by
        - first (optional) computing differential extinctions
        - adding linear components of the lensed source(s)
        - adding linear components of the unlensed components (i.e. deflector)
        - adding point sources (can be multiple lensed or stars in the field)

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: bool, if True, computes components without convolution kernel (will not work for point sources)
        :return: response matrix (m x n)
        """
        x_grid, y_grid = self.ImageNumerics.coordinates_evaluate
        source_light_response, n_source = self.source_mapping.image_flux_split(
            x_grid, y_grid, kwargs_lens, kwargs_source
        )
        extinction = self._extinction.extinction(
            x_grid,
            y_grid,
            kwargs_extinction=kwargs_extinction,
            kwargs_special=kwargs_special,
        )
        lens_light_response, n_lens_light = self.LensLightModel.functions_split(
            x_grid, y_grid, kwargs_lens_light
        )

        ra_pos, dec_pos, amp, n_points = self.point_source_linear_response_set(
            kwargs_ps, kwargs_lens, kwargs_special, with_amp=False
        )
        num_param = n_points + n_lens_light + n_source

        num_response = self.num_data_evaluate
        A = np.zeros((num_param, num_response))
        n = 0
        # response of lensed source profile
        for i in range(0, n_source):
            image = source_light_response[i]

            # multiply with primary beam before convolution
            if self._pb is not None:
                image *= self._pb_1d

            image *= extinction
            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = np.nan_to_num(self.image2array_masked(image), copy=False)
            n += 1
        # response of deflector light profile (or any other un-lensed extended components)
        for i in range(0, n_lens_light):
            image = lens_light_response[i]

            # multiply with primary beam before convolution
            if self._pb is not None:
                image *= self._pb_1d

            image = self.ImageNumerics.re_size_convolve(image, unconvolved=unconvolved)
            A[n, :] = np.nan_to_num(self.image2array_masked(image), copy=False)
            n += 1
        # response of point sources
        for i in range(0, n_points):
            # raise warnings when primary beam is attempted to be applied for point sources
            if self._pb is not None:
                raise Warning("Antenna primary beam does not apply to point sources!")

            image = self.ImageNumerics.point_source_rendering(
                ra_pos[i], dec_pos[i], amp[i]
            )
            A[n, :] = np.nan_to_num(self.image2array_masked(image), copy=False)
            n += 1
        return A * self._flux_scaling

    def update_linear_kwargs(
        self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    ):
        """Links linear parameters to kwargs arguments.

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        return self._update_linear_kwargs(
            param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )

    def _update_linear_kwargs(
        self, param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    ):
        """Links linear parameters to kwargs arguments.

        :param param: linear parameter vector corresponding to the response matrix
        :return: updated list of kwargs with linear parameter values
        """
        i = 0
        kwargs_source, i = self.SourceModel.update_linear(
            param, i, kwargs_list=kwargs_source
        )
        kwargs_lens_light, i = self.LensLightModel.update_linear(
            param, i, kwargs_list=kwargs_lens_light
        )
        kwargs_ps, i = self.PointSource.update_linear(param, i, kwargs_ps, kwargs_lens)
        return kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps

    def linear_param_from_kwargs(self, kwargs_source, kwargs_lens_light, kwargs_ps):
        """Inverse function of update_linear() returning the linear amplitude list for
        the keyword argument list.

        :param kwargs_source: 
        :param kwargs_lens_light: 
        :param kwargs_ps: 
        :return: list of linear coefficients
        """
        return self._linear_param_from_kwargs(
            kwargs_source, kwargs_lens_light, kwargs_ps
        )

    def _linear_param_from_kwargs(self, kwargs_source, kwargs_lens_light, kwargs_ps):
        """Inverse function of update_linear() returning the linear amplitude list for
        the keyword argument list.

        :param kwargs_source: 
        :param kwargs_lens_light: 
        :param kwargs_ps: 
        :return: list of linear coefficients
        """
        param = []
        param += self.SourceModel.linear_param_from_kwargs(kwargs_source)
        param += self.LensLightModel.linear_param_from_kwargs(kwargs_lens_light)
        param += self.PointSource.linear_param_from_kwargs(kwargs_ps)
        return param

    def update_pixel_kwargs(self, kwargs_source, kwargs_lens_light):
        """Update kwargs arguments for pixel-based profiles with fixed properties such
        as their number of pixels, scale, and center coordinates (fixed to the origin).

        :param kwargs_source: list of keyword arguments corresponding to the
                superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to the
                superposition of different lens light profiles
        :return: updated kwargs_source and kwargs_lens_light
        """
        # in case the source plane grid size has changed, update the kwargs accordingly
        ss_factor_source = self.SourceNumerics.grid_supersampling_factor
        kwargs_source[0]["n_pixels"] = int(
            self.Data.num_pixel * ss_factor_source**2
        )  #  effective number of pixels in source plane
        kwargs_source[0]["scale"] = (
            self.Data.pixel_width / ss_factor_source
        )  # effective pixel size of source plane grid
        # pixelated reconstructions have no well-defined center, we put it arbitrarily at (0, 0), center of the image
        kwargs_source[0]["center_x"] = 0
        kwargs_source[0]["center_y"] = 0
        # do the same if the lens light has been reconstructed
        if kwargs_lens_light is not None and len(kwargs_lens_light) > 0:
            kwargs_lens_light[0]["n_pixels"] = self.Data.num_pixel
            kwargs_lens_light[0]["scale"] = self.Data.pixel_width
            kwargs_lens_light[0]["center_x"] = 0
            kwargs_lens_light[0]["center_y"] = 0
        return kwargs_source, kwargs_lens_light

    def reduced_residuals(self, model, error_map=0):
        """:param model: 2d numpy array of the modeled image :param error_map: 2d numpy
        array of additional noise/error terms from model components (such as PSF model
        uncertainties) :return: 2d numpy array of reduced residuals per pixel."""
        mask = self.likelihood_mask
        C_D = self.Data.C_D_model(model)
        residual = (model - self.Data.data) / np.sqrt(C_D + np.abs(error_map)) * mask
        return residual

    def reduced_chi2(self, model, error_map=0):
        """Returns reduced chi2 :param model: 2d numpy array of a model predicted image
        :param error_map: same format as model, additional error component (such as PSF
        errors) :return: reduced chi2."""
        norm_res = self.reduced_residuals(model, error_map)
        return np.sum(norm_res**2) / self.num_data_evaluate

    @property
    def num_data_evaluate(self):
        """Number of data points to be used in the linear solver :return: number of
        evaluated data points :rtype: int."""
        return int(np.sum(self.likelihood_mask))

    def update_data(self, data_class):
        """:param data_class: instance of Data() class :return: no return.

        Class is updated.
        """
        self.Data = data_class
        self.ImageNumerics._PixelGrid = data_class

    def image2array_masked(self, image):
        """Returns 1d array of values in image that are not masked out for the
        likelihood computation/linear minimization :param image: 2d numpy array of full
        image :return: 1d array."""
        array = util.image2array(image)
        return array[self._mask1d]

    def array_masked2image(self, array):
        """:param array: 1d array of values not masked out (part of linear fitting)
        :return: 2d array of full image."""
        nx, ny = self.Data.num_pixel_axes
        grid1d = np.zeros(nx * ny)
        grid1d[self._mask1d] = array
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    def _error_map_model(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Noise estimate (variances as diagonal of the pixel covariance matrix)
        resulted from inherent model uncertainties This term is currently the psf error
        map.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_ps: point source keyword arguments
        :param kwargs_special: special parameter keyword arguments
        :return: 2d array corresponding to the pixels in terms of variance in noise
        """
        return self._error_map_psf(kwargs_lens, kwargs_ps, kwargs_special)

    def _error_map_psf(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Map of image with error terms (sigma**2) expected from inaccuracies in the
        PSF modeling.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_ps: point source keyword arguments
        :param kwargs_special: special parameter keyword arguments
        :return: 2d array of size of the image
        """
        error_map = np.zeros(self.Data.num_pixel_axes)
        if self._psf_error_map is True:
            for k, bool_ in enumerate(self._psf_error_map_bool_list):
                if bool_ is True:
                    ra_pos, dec_pos, _ = self.PointSource.point_source_list(
                        kwargs_ps, kwargs_lens=kwargs_lens, k=k, with_amp=False
                    )
                    if len(ra_pos) > 0:
                        ra_pos, dec_pos = self._displace_astrometry(
                            ra_pos, dec_pos, kwargs_special=kwargs_special
                        )
                        error_map += self.ImageNumerics.psf_error_map(
                            ra_pos,
                            dec_pos,
                            None,
                            self.Data.data,
                            fix_psf_error_map=False,
                        )
        return error_map

    def error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """Variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source
        reconstruction as a sum of the errors of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """
        return self._error_map_source(kwargs_source, x_grid, y_grid, cov_param)

    def _error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """Variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source
        reconstruction as a sum of the errors of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """

        error_map = np.zeros_like(x_grid)
        basis_functions, n_source = self.SourceModel.functions_split(
            x_grid, y_grid, kwargs_source
        )
        basis_functions = np.array(basis_functions)

        if cov_param is not None:
            for i in range(len(error_map)):
                error_map[i] = (
                    basis_functions[:, i]
                    .T.dot(cov_param[:n_source, :n_source])
                    .dot(basis_functions[:, i])
                )
        return error_map

    def point_source_linear_response_set(
        self, kwargs_ps, kwargs_lens, kwargs_special, with_amp=True
    ):
        """:param kwargs_ps: point source keyword argument list :param kwargs_lens: lens
        model keyword argument list :param kwargs_special: special keyword argument
        list, may include 'delta_x_image' and 'delta_y_image' :param with_amp: bool, if
        True, relative magnification between multiply imaged point sources are held
        fixed.

        :return: list of positions and amplitudes split in different basis components
                with applied astrometric corrections
        """

        ra_pos, dec_pos, amp, n_points = self.PointSource.linear_response_set(
            kwargs_ps, kwargs_lens, with_amp=with_amp
        )

        if kwargs_special is not None:
            if "delta_x_image" in kwargs_special:
                delta_x, delta_y = (
                    kwargs_special["delta_x_image"],
                    kwargs_special["delta_y_image"],
                )
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

    def check_positive_flux(self, kwargs_source, kwargs_lens_light, kwargs_ps):
        """Checks whether the surface brightness profiles contain positive fluxes and
        returns bool if True.

        :param kwargs_source: source surface brightness keyword argument list
        :param kwargs_lens_light: lens surface brightness keyword argument list
        :param kwargs_ps: point source keyword argument list
        :return: boolean
        """
        pos_bool_ps = self.PointSource.check_positive_flux(kwargs_ps)
        if self._pixelbased_bool is True:
            # this constraint must be handled by the pixel-based solver
            pos_bool_source = True
            pos_bool_lens_light = True
        else:
            pos_bool_source = self.SourceModel.check_positive_flux_profile(
                kwargs_source
            )
            pos_bool_lens_light = self.LensLightModel.check_positive_flux_profile(
                kwargs_lens_light
            )
        if (
            pos_bool_ps is True
            and pos_bool_source is True
            and pos_bool_lens_light is True
        ):
            return True
        else:
            return False

    # linear solver for interferometric natwt method
    def _image_linear_solve_interferometry_natwt(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
    ):
        """'interferometry_natwt' method does NOT support model_error, cov_param. The
        interferometry linear solver just does the linear solving to get the optimal
        linear amplitudes and apply the marginalized amplitudes to make the model
        images.

        :return: model, model_error, cov_param, param
        model and param are the same returns of self._image_linear_solve_interferometry_natwt_solving(A, d) function
        model_error =0 and cov_param = None for the interferometric method.
        """
        A = self._linear_response_matrix(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            kwargs_extinction,
            kwargs_special,
            unconvolved=True,
        )
        d = self.data_response
        model, param = self._image_linear_solve_interferometry_natwt_solving(A, d)
        model_error = 0  # just a place holder
        cov_param = None  # just a place holder
        _, _, _, _ = self._update_linear_kwargs(
            param, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )
        return model, model_error, cov_param, param

    def _image_linear_solve_interferometry_natwt_solving(self, A, d):
        """Linearly solve the amplitude of each light profile response to the natural
        weighting interferometry images, based on (placeholder for Nan Zhang's paper).

        Theories:
            Suppose there are a set of light responses :math:`\\{x_i\\}`, we want to solve the set of amplitudes :math:`\\{\\alpha_i\\}`,
            such that minimizes the chi^2 given by
            .. math::
                    \\chi^2 = (d - A_{PSF}\\sum_i \\alpha_i x_i)^TC^{-1}(d - A_{PSF}\\sum_i \\alpha_i x_i),
            where :math:`A_{PSF}` is the PSF convolution operation matrix (not to be confused with the input A of this function)
            and :math:`C` is the noise covariance matrix. :math:`d` is the data image.
            For natural weighting interferometric images, we have :math:`C = \\sigma^2 A_{PSF}`,
            (see Section 3.2 of https://doi.org/10.1093/mnras/staa2740 for the relation of natural weighting covariance matrix and PSF convolution)
            therefore the chi^2 function simplifies to
            .. math::
                    \\chi^2 = \\frac{1}{\\sigma^2}(d^TA_{PSF}^{-1}d + \\sum_{i,j}\\alpha_i\\alpha_j x_i^TA_{PSF}x_j - 2\\sum_{i}x_i^Td),
            from which the optimal amplitudes :math:`\\{\\alpha_i\\}` can be solved linearly by solving
            .. math::
                    \\sum_{j} M_{ij}\\alpha_{j} = b_i,
            where :math:`M_{ij} = \\frac{1}{\\sigma^2}x_i^TA_{PSF}x_j` and :math:`b_{i} = \\frac{1}{\\sigma^2}x_i^Td`.

        The steps of this function are:
            (1.) Making the entries :math:`M_{ij}` and :math:`b_i` defined above.
            (2.) Solve the linear function to get the optimal amplitudes.
            (3.) Apply these optimal amplitudes to make unconvolved and convolved model images.
                The output model images are in the form [array1, array2].
                (Note that this is different from the non-interferometric linear solver of Lenstronomy,
                 this output form saves time for likelihood computations in imaging_data for interferometric method.)
                array1 is the unconvolved model image :math:`array1 = \\sum_i \\alpha_i x_i`, where :math:`\\alpha_i` is the solved optimal amplitudes.
                array2 is the convolved model image :math:`array2 = A_{PSF}\\sum_i \\alpha_i x_i`, where :math:`\\alpha_i`.

        :param A: response of unconvolved light profiles, [x_1, x_2, ...]
        :param d: data image, d
        :return: [array1, array2], [amp_array]
        where the [array1, array2] are unconvolved and convolved model images with solved amplitudes
        and [amp_array] are the solved optimal amplitudes.
        """
        num_of_light, num_of_image_pixel = np.shape(A)

        A_convolved = np.zeros(np.shape(A))

        # convolve each response separately
        for i in range(num_of_light):
            A_convolved[i] = util.image2array(
                self._convolution._static_fft(util.array2image(A[i]), mode="same")
            )

        M = np.zeros((num_of_light, num_of_light))
        for i in range(num_of_light):
            for j in range(num_of_light):
                if j < i:
                    M[i, j] = M[j, i]
                else:
                    M[i, j] = np.sum(A_convolved[j] * A[i])

        b = np.zeros((num_of_light))
        for i in range(num_of_light):
            b[i] = np.sum(A[i] * (d))

        param_amps = np.linalg.lstsq(M, b)[0]

        clean_temp = np.zeros((num_of_image_pixel))
        dirty_temp = np.zeros((num_of_image_pixel))
        for i in range(num_of_light):
            clean_temp += param_amps[i] * A[i]
            dirty_temp += param_amps[i] * A_convolved[i]

        clean_model = util.array2image(clean_temp)
        dirty_model = util.array2image(dirty_temp)

        model = [clean_model, dirty_model]

        return model, param_amps
