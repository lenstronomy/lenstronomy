__author__ = "dangilman"

from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane

__all__ = ["MultiPlaneDecoupled"]


class MultiPlaneDecoupled(MultiPlane):

    def __init__(self,
                 z_source,
                 lens_model_list,
                 lens_redshift_list,
                 cosmo=None,
                 numerical_alpha_class=None,
                 observed_convention_index=None,
                 ignore_observed_positions=False,
                 z_source_convention=None,
                 cosmo_interp=False,
                 z_interp_stop=None,
                 num_z_interp=100,
                 kwargs_interp=None,
                 kwargs_synthesis=None,
                 x0_interp=None,
                 y0_interp=None,
                 alpha_x_interp_foreground=None,
                 alpha_y_interp_foreground=None,
                 alpha_x_interp_background=None,
                 alpha_y_interp_background=None,
                 z_split=None,
                 ):
        """
        A class for multiplane lensing in which the deflection angles at certain coordinates are fixed through
        usser-specified interpolation functions. These functions return fixed deflection angles that effectively
        decouple deflections by a group of deflectors at redshift Z from deflections produced by halos at redshift< Z.

        This class effectively breaks the recursive nature of the multi-plane lens equation, and can significantly speed
        up computations with a large number of line-of-sight halos.

        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
         models. If None, 'z_source' is used.
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
         (see documentation in Profiles/numerical_alpha)
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.
        :param kwargs_synthesis: keyword arguments for the 'SYNTHESIS' lens model, if applicable
        :param x0_interp: a function that maps an angular coordinate on the sky to the x coordinate of a physical
        position [Mpc] at the first lens plane
        :param y0_interp: same as x0_interp, but returns the y coordinate in Mpc
        :param alpha_x_interp_list: a list of functions that take as input angular coordinates (x, y) and returns the
        x-component of the deflection angle at each coorindate
        :param alpha_y_interp_list: same as alpha_x_interp_list, but returns the y-component of the deflection angle
        at (x,y)
        :param z_interp_list: a list of redshifts corresponding to the alpha_x_interp_list and alpha_y_interp_list
        entries
        """
        self._alphax_interp_foreground = alpha_x_interp_foreground
        self._alphay_interp_foreground = alpha_y_interp_foreground
        self._alphax_interp_background = alpha_x_interp_background
        self._alphay_interp_background = alpha_y_interp_background
        self._x0_interp = x0_interp
        self._y0_interp = y0_interp
        self._z_split = z_split
        super(MultiPlaneDecoupled, self).__init__(
            z_source,
            lens_model_list,
            lens_redshift_list,
            cosmo,
            numerical_alpha_class,
            observed_convention_index,
            ignore_observed_positions,
            z_source_convention,
            cosmo_interp,
            z_interp_stop,
            num_z_interp,
            kwargs_interp,
            kwargs_synthesis
        )
        cosmo_bkg = self._multi_plane_base._cosmo_bkg
        d_xy_source = cosmo_bkg.d_xy(0, z_source)
        d_xy_lens_source = cosmo_bkg.d_xy(self._z_split, z_source)
        self._reduced_to_phys = d_xy_source / d_xy_lens_source
        self._deltaT_zsplit = cosmo_bkg.d_xy(0, self._z_split)
        self._deltaT_zsplit_zsource = cosmo_bkg.T_xy(self._z_split, z_source)
        self._Ts = cosmo_bkg.T_xy(0, z_source)
        self._Td = cosmo_bkg.T_xy(0, z_split)
        self._Tds = cosmo_bkg.T_xy(self._z_split, z_source)

    def geo_shapiro_delay(*args, **kwargs
    ):

        raise Exception('time delays are not yet implemented for the MultiPlaneDecoupled class')

    def ray_shooting_partial(
        self,
            *args, **kwargs
    ):
        raise Exception('ray_shooting_partial is not well-defined for this class')

    def ray_shooting(
            self,
            theta_x,
            theta_y,
            kwargs_lens,
            *args, **kwargs
    ):
        """Ray-shooting through the lens volume with fixed deflection angles at certain lens planes passed through
         the alpha_x_interp/alpha_y_interp lists. Starts with (x,y) co-moving
        distance passed through the x0_interp and y0_interp functions, then starts multi-plane ray-tracing through all
        subsequent lens planes.

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :return: co-moving position and angles at redshift z_stop
        """

        # first we map to the first lens plane; this overwrites x and y passed into the function.....
        coordinates = (theta_x, theta_y)
        x = self._x0_interp(coordinates)
        y = self._y0_interp(coordinates)

        # compute the deflections up to z_split
        alpha_x_foreground = self._alphax_interp_foreground(coordinates)
        alpha_y_foreground = self._alphay_interp_foreground(coordinates)

        # now we compute the deflection angles at the first redshift plane for the interpolated deflection fields
        alpha_beta_subx = self._alphax_interp_background(coordinates)
        alpha_beta_suby = self._alphay_interp_background(coordinates)

        # now we compute the deflection angles for the lens model(s) corresponding to kwargs_lens
        alpha_x_main = 0.0
        alpha_y_main = 0.0
        theta_x_main = x / self._Td
        theta_y_main = y / self._Td
        # print(len(kwargs_lens))
        # print(kwargs_lens)
        # print(len(self._multi_plane_base.func_list))
        # a=input('continue')
        for k in range(0, len(kwargs_lens)):
            alpha_x_reduced, alpha_y_reduced = self._multi_plane_base.func_list[k].derivatives(
                theta_x_main, theta_y_main, **kwargs_lens[k])
            alpha_x_main += self._reduced_to_phys * alpha_x_reduced
            alpha_y_main += self._reduced_to_phys * alpha_y_reduced
        alpha_x, alpha_y = alpha_x_foreground - alpha_x_main, alpha_y_foreground - alpha_y_main

        # alpha_beta_subx = source_x * Ts / Tds - xD / Tds - alpha_x
        beta_x = alpha_beta_subx * self._Tds / self._Ts + x / self._Ts + alpha_x * self._Tds / self._Ts
        beta_y = alpha_beta_suby * self._Tds / self._Ts + y / self._Ts + alpha_y * self._Tds / self._Ts
        return beta_x, beta_y
