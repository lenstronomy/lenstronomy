import numpy as np
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.MultiPlane.multi_plane_base import MultiPlaneBase
import lenstronomy.Util.constants as const

__all__ = ["MultiPlaneBase"]


class MultiPlaneDecoupled(MultiPlaneBase):
    def __init__(
        self,
        lens_model_list,
        lens_redshift_list,
        z_source_convention,
        cosmo=None,
        numerical_alpha_class=None,
        cosmo_interp=False,
        z_interp_stop=None,
        num_z_interp=100,
        kwargs_interp=None,
        kwargs_synthesis=None,
        alpha_x_interp_list=None,
        alpha_y_interp_list=None,
        z_interp_list=None,
    ):
        """A class for multiplane lensing in which the deflection angles at certain
        coordinates are fixed through usser-specified interpolation functions. These
        functions return fixed deflection angles that effectively decouple deflections
        by a group of deflectors at redshift Z from deflections produced by halos at
        redshift< Z.

        This class effectively breaks the recursive nature of the multi-plane lens
        equation, and can significantly speed up computations with a large number of
        line-of-sight halos.

        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models
            indicated in lens_model_list
        :param z_source_convention: float, redshift of a source to define the reduced
            deflection angles of the lens models. If None, 'z_source' is used.
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in
            NumericalAlpha() lens model (see documentation in Profiles/numerical_alpha)
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
            See description in the Interpolate() class. Only applicable for 'INTERPOL'
            and 'INTERPOL_SCALED' models.
        :param kwargs_synthesis: keyword arguments for the 'SYNTHESIS' lens model, if
            applicable
        :param alpha_x_interp_list: a list of functions that take as input angular
            coordinates (x, y) and returns the x-component of the deflection angle at
            each coorindate
        :param alpha_y_interp_list: same as alpha_x_interp_list, but returns the
            y-component of the deflection angle at (x,y)
        :param z_interp_list: a list of redshifts corresponding to the
            alpha_x_interp_list and alpha_y_interp_list entries
        """

        self._alphax_interp = alpha_x_interp_list
        self._alphay_interp = alpha_y_interp_list
        self._z_interp_list = z_interp_list
        super(MultiPlaneDecoupled, self).__init__(
            lens_model_list,
            lens_redshift_list,
            z_source_convention,
            cosmo,
            numerical_alpha_class,
            cosmo_interp,
            z_interp_stop,
            num_z_interp,
            kwargs_interp,
            kwargs_synthesis,
        )
        self._fixed_deflection_plane = [True] * len(z_interp_list) + [False] * len(
            self._lens_redshift_list
        )
        self._lens_redshift_list_complete = z_interp_list + self._lens_redshift_list
        self._sorted_redshift_indexes = self._index_ordering(
            self._lens_redshift_list_complete
        )

    def geo_shapiro_delay(
        self, theta_x, theta_y, kwargs_lens, z_stop, T_z_stop=None, T_ij_end=None
    ):
        raise Exception(
            "time delays are not yet implemented for the MultiPlaneDecoupled class"
        )

    def ray_shooting_partial(
        self,
        x,
        y,
        alpha_x,
        alpha_y,
        z_start,
        z_stop,
        kwargs_lens,
        *args,
        **kwargs,
    ):
        """Ray-shooting through the lens volume with fixed deflection angles at certina
        lens planes passed through the alpha_x_interp/alpha_y_interp lists. Starts with
        (x,y) co-moving distance passed through the x0_interp and y0_interp functions,
        then starts multi-plane ray-tracing through all subsequent lens planes.

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :return: co-moving position and angles at redshift z_stop
        """

        z_lens_last = z_start
        counter_kwargs_lens = 0
        counter_fixed_deflection_plane = 0
        # NOTE: currently the fixed alpha interpolations have to be specified in order of ascending redshift
        for i, idex in enumerate(self._sorted_redshift_indexes):
            z_lens = self._lens_redshift_list_complete[idex]
            if z_lens <= z_stop:
                delta_T = self._cosmo_bkg.T_xy(z_start, z_lens)
                x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
                # the difference between this class and MultiPlaneBase is here
                if self._fixed_deflection_plane[i]:
                    angle_at_x = x / delta_T
                    angle_at_y = y / delta_T
                    # these deflection angles are only functions of the angular position where the ray intersects
                    # the lens plane and DO NOT depend on alpha_x, alpha_y, the incoming ray angle. This effectively
                    # de-couples these lens models from the rest of the lens system
                    alpha_x = self._alphax_interp[counter_fixed_deflection_plane](
                        angle_at_x, angle_at_y
                    )
                    alpha_y = self._alphay_interp[counter_fixed_deflection_plane](
                        angle_at_x, angle_at_y
                    )
                    counter_fixed_deflection_plane += 1
                else:
                    alpha_x, alpha_y = self._add_deflection(
                        x, y, alpha_x, alpha_y, kwargs_lens, counter_kwargs_lens
                    )
                    counter_kwargs_lens += 1
                alpha_x, alpha_y = self._add_deflection(
                    x, y, alpha_x, alpha_y, kwargs_lens, i
                )
                z_lens_last = z_lens
        if z_lens_last == z_stop:
            delta_T = 0
        else:
            delta_T = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
        x, y = self._ray_step_add(x, y, alpha_x, alpha_y, delta_T)
        return x, y, alpha_x, alpha_y
