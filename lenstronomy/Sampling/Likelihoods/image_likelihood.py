import numpy as np
from lenstronomy.Util import class_creator
from lenstronomy.Util import util

__all__ = ["ImageLikelihood"]


class ImageLikelihood(object):
    """Manages imaging data likelihoods."""

    def __init__(
        self,
        multi_band_list,
        multi_band_type,
        kwargs_model,
        bands_compute=None,
        image_likelihood_mask_list=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        kwargs_pixelbased=None,
        linear_solver=True,
    ):
        """

        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the
         fitting
        :param image_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be
         evaluated in the likelihood
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the inferred
         linear coefficients
        :param linear_prior: float or list of floats (when multi-linear setting is chosen) indicating the range of
         linear amplitude priors when computing the marginalization term.
        :param check_positive_flux: bool, option to punish models that do not have all positive linear amplitude
         parameters
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver
         (see SLITronomy documentation)
        :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        """
        self.imSim = class_creator.create_im_sim(
            multi_band_list,
            multi_band_type,
            kwargs_model,
            bands_compute=bands_compute,
            image_likelihood_mask_list=image_likelihood_mask_list,
            kwargs_pixelbased=kwargs_pixelbased,
            linear_solver=linear_solver,
        )
        self._model_type = self.imSim.type
        self._source_marg = source_marg
        self._linear_prior = linear_prior
        self._check_positive_flux = check_positive_flux
        self.num_bands = len(multi_band_list)

    def logL(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_special=None,
        kwargs_extinction=None,
        **kwargs,
    ):
        """

        :param kwargs_lens: lens model keyword argument list according to LensModel module
        :param kwargs_source: source light keyword argument list according to LightModel module
        :param kwargs_lens_light: deflector light (not lensed) keyword argument list according to LightModel module
        :param kwargs_ps: point source keyword argument list according to PointSource module
        :param kwargs_special: special keyword argument list as part of the Param module
        :param kwargs_extinction: extinction parameter keyword argument list according to LightModel module
        :return: log likelihood of the data given the model, linear parameter inversion list
        """

        backup_grids = self._apply_multiband_offsets(kwargs_special)
        try:
            logL, param = self.imSim.likelihood_data_given_model(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction=kwargs_extinction,
                kwargs_special=kwargs_special,
                source_marg=self._source_marg,
                linear_prior=self._linear_prior,
                check_positive_flux=self._check_positive_flux,
            )

            if np.isnan(logL) is True:
                return -(10**15), param
            
        finally:
            self._restore_multiband_offsets(backup_grids)
        
        return logL, param

    @property
    def num_data(self):
        """

        :return: number of image data points
        """
        return self.imSim.num_data_evaluate

    def num_param_linear(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_special=None,
        kwargs_extinction=None,
        kwargs_tracer_source=None,
    ):
        """

        :return:  number of linear parameters solved for during the image reconstruction process
        """
        return self.imSim.num_param_linear(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
        )

    def reset_point_source_cache(self, cache=True):
        """

        :param cache: boolean
        :return: None
        """
        self.imSim.reset_point_source_cache(cache=cache)

    def _apply_multiband_offsets(self, kwargs_special):
            """
            Temporarily apply model-side astrometric corrections (shift and rotation) to the coordinate grid.

            The sampled offsets represent corrections applied to the model coordinate system during likelihood evaluation.
            Therefore, when interpreting recovered parameters as astrometric offsets between observed bands, the sign is
            opposite.

            :param kwargs_special: contains 'kwargs_offsets', where each dictionary specifies the correction for the corresponding band.
            :return: dict; backup of the original coordinate states to be restored after logL evaluation
            """
            if kwargs_special is None or 'kwargs_offsets' not in kwargs_special:
                return None

            kwargs_offsets = kwargs_special['kwargs_offsets']
            backup_grids = {}

            for band_index, offset in enumerate(kwargs_offsets):
                if not offset:  # Skip if dict is empty or None
                    continue
                if band_index >= self.num_bands:
                    break
                
                dx = offset.get('dx', 0)
                dy = offset.get('dy', 0)
                angle = offset.get('angle', 0)

                if dx == 0 and dy == 0 and angle == 0:
                    continue
                
                # Get the list containing SingleBandMultiModel objects
                if hasattr(self.imSim, '_imageModel_list'):
                    image_model_list = self.imSim._imageModel_list
                elif hasattr(self.imSim, 'imageModel_list'):
                    image_model_list = self.imSim.imageModel_list
                else:
                    image_model_list = [self.imSim] # single band case

                # Backup original state
                data_class = image_model_list[band_index].Data
                grid = image_model_list[band_index].ImageNumerics._numerics_subframe._grid

                backup_grids[band_index] = {
                    'data_ra_at_xy_0': data_class._ra_at_xy_0,
                    'data_dec_at_xy_0': data_class._dec_at_xy_0,
                    'data_Mpix2a': np.copy(data_class._Mpix2a),

                    'grid_ra_at_xy_0': grid._ra_at_xy_0,
                    'grid_dec_at_xy_0': grid._dec_at_xy_0,
                    'grid_Mpix2a': np.copy(grid._Mpix2a),
                    'grid_Ma2pix': np.copy(grid._Ma2pix),
                    'grid_x_grid': np.copy(grid._x_grid),
                    'grid_y_grid': np.copy(grid._y_grid),
                    'grid_ra_subgrid': np.copy(grid._ra_subgrid),
                    'grid_dec_subgrid': np.copy(grid._dec_subgrid),
                }

                # Calculate rotation around the geometric center
                nx, ny = data_class._nx, data_class._ny
                cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
                M_old = data_class._Mpix2a
                ra_0_old = data_class._ra_at_xy_0
                dec_0_old = data_class._dec_at_xy_0

                ra_center = ra_0_old + M_old[0, 0] * cx + M_old[0, 1] * cy
                dec_center = dec_0_old + M_old[1, 0] * cx + M_old[1, 1] * cy

                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], 
                                    [sin_a,  cos_a]])
                M_new = np.dot(rot_matrix, M_old)

                ra_0_new = ra_center - (M_new[0, 0] * cx + M_new[0, 1] * cy) + dx
                dec_0_new = dec_center - (M_new[1, 0] * cx + M_new[1, 1] * cy) + dy

                # Mutate internal attributes
                data_class._ra_at_xy_0 = ra_0_new
                data_class._dec_at_xy_0 = dec_0_new
                data_class._Mpix2a = M_new
                
                # Force grid re-evaluation
                x_grid, y_grid = data_class.coordinate_grid(nx, ny)
                data_class._x_grid = np.atleast_1d(x_grid)
                data_class._y_grid = np.atleast_1d(y_grid)

                # Update RegularGrid cache used by ImageNumerics
                grid = image_model_list[band_index].ImageNumerics._numerics_subframe._grid

                grid._ra_at_xy_0 = data_class._ra_at_xy_0
                grid._dec_at_xy_0 = data_class._dec_at_xy_0
                grid._Mpix2a = data_class._Mpix2a
                grid._Ma2pix = data_class._Ma2pix

                grid._x_grid, grid._y_grid = grid.coordinate_grid(
                    grid._nx,
                    grid._ny,
                )

                x_sub, y_sub = util.make_subgrid(
                    grid._x_grid,
                    grid._y_grid,
                    grid._supersampling_factor,
                )

                grid._ra_subgrid = x_sub[grid._compute_indexes]
                grid._dec_subgrid = y_sub[grid._compute_indexes]

            return backup_grids

    def _restore_multiband_offsets(self, backup_grids):
        """
        Restore the original coordinate grids after logL evaluation.

        :param backup_grids: dict; coordinate states returned by _apply_multiband_offsets
        """
        if backup_grids is None:
            return

        # Get the list containing SingleBandMultiModel objects
        if hasattr(self.imSim, '_imageModel_list'):
            image_model_list = self.imSim._imageModel_list
        elif hasattr(self.imSim, 'imageModel_list'):
            image_model_list = self.imSim.imageModel_list
        else:
            image_model_list = [self.imSim] # single band case

        for band_index, backup in backup_grids.items():
            data_class = image_model_list[band_index].Data

            data_class._ra_at_xy_0 = backup['data_ra_at_xy_0']
            data_class._dec_at_xy_0 = backup['data_dec_at_xy_0']
            data_class._Mpix2a = backup['data_Mpix2a']

            grid = image_model_list[band_index].ImageNumerics._numerics_subframe._grid

            grid._ra_at_xy_0 = backup['grid_ra_at_xy_0']
            grid._dec_at_xy_0 = backup['grid_dec_at_xy_0']
            grid._Mpix2a = backup['grid_Mpix2a']
            grid._Ma2pix = backup['grid_Ma2pix']

            grid._x_grid = backup['grid_x_grid']
            grid._y_grid = backup['grid_y_grid']

            grid._ra_subgrid = backup['grid_ra_subgrid']
            grid._dec_subgrid = backup['grid_dec_subgrid']