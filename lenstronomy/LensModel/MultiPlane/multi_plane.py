import numpy as np
from copy import deepcopy
from lenstronomy.LensModel.MultiPlane.multi_plane_base import MultiPlaneBase

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class MultiPlane(object):
    """Multi-plane lensing class with option to assign positions of a selected set of
    lens models in the observed plane.

    The lens model deflection angles are in units of reduced deflections from the
    specified redshift of the lens to the source redshift of the class instance.
    """

    def __init__(
        self,
        z_source,
        lens_model_list,
        lens_redshift_list,
        cosmo=None,
        numerical_alpha_class=None,
        observed_convention_index=None,
        ignore_observed_positions=False,
        z_source_convention=None,
        z_lens_convention=None,
        cosmo_interp=False,
        z_interp_stop=None,
        num_z_interp=100,
        kwargs_interp=None,
        kwargs_synthesis=None,
        distance_ratio_sampling=False,
    ):
        """

        :param z_source: source redshift for default computation of reduced lensing quantities
        :param lens_model_list: list of lens model strings
        :param lens_redshift_list: list of floats with redshifts of the lens models indicated in lens_model_list
        :param cosmo: instance of astropy.cosmology
        :param numerical_alpha_class: an instance of a custom class for use in NumericalAlpha() lens model
         (see documentation in Profiles/numerical_alpha)
        :param kwargs_interp: interpolation keyword arguments specifying the numerics.
         See description in the Interpolate() class. Only applicable for 'INTERPOL' and 'INTERPOL_SCALED' models.
        :param observed_convention_index: a list of indices, corresponding to the lens_model_list element with same
         index, where the 'center_x' and 'center_y' kwargs correspond to observed (lensed) positions, not physical
         positions. The code will compute the physical locations when performing computations
        :param ignore_observed_positions: bool, if True, will ignore the conversion between observed to physical
         position of deflectors
        :param z_source_convention: float, redshift of a source to define the reduced deflection angles of the lens
        models. If None, 'z_source' is used.
        :param z_lens_convention: float, redshift of a lens plane to define the
        effective time-delay distance. Only needed if distance ratios are
        sampled. If None, the first lens redshift is used.
        :param kwargs_synthesis: keyword arguments for the 'SYNTHESIS' lens model, if applicable
        :param distance_ratio_sampling: bool, if True, will use sampled
        distance ratios to update T_ij value in multi-lens plane computation.
        """

        if z_source_convention is None:
            z_source_convention = z_source
        if z_interp_stop is None:
            z_interp_stop = max(z_source, z_source_convention)
        if z_interp_stop < max(z_source, z_source_convention):
            raise ValueError(
                "z_interp_stop= %s needs to be larger or equal the maximum of z_source=%s and "
                "z_source_convention=%s"
                % (z_interp_stop, z_source, z_source_convention)
            )
        self._z_source_convention = z_source_convention
        if z_lens_convention is None:
            if len(lens_redshift_list) > 0:
                self._z_lens_convention = np.min(lens_redshift_list)
            else:
                self._z_lens_convention = 0
        else:
            self._z_lens_convention = z_lens_convention
        self.distance_ratio_sampling = distance_ratio_sampling
        self._multi_plane_base = MultiPlaneBase(
            lens_model_list=lens_model_list,
            lens_redshift_list=lens_redshift_list,
            cosmo=cosmo,
            numerical_alpha_class=numerical_alpha_class,
            z_source_convention=z_source_convention,
            cosmo_interp=cosmo_interp,
            z_interp_stop=z_interp_stop,
            num_z_interp=num_z_interp,
            kwargs_interp=kwargs_interp,
            kwargs_synthesis=kwargs_synthesis,
        )

        self._set_source_distances(z_source)
        self._observed_convention_index = observed_convention_index
        if observed_convention_index is None:
            self._convention = PhysicalLocation()
        else:
            assert isinstance(observed_convention_index, list)
            self._convention = LensedLocation(
                self._multi_plane_base, observed_convention_index
            )
        self.ignore_observed_positions = ignore_observed_positions

    def update_source_redshift(self, z_source):
        """Update instance of this class to compute reduced lensing quantities and time
        delays to a specific source redshift.

        :param z_source: float; source redshift
        :return: self variables update to new redshift
        """
        if z_source == self._z_source:
            pass
        else:
            self._set_source_distances(z_source)

    @property
    def multi_plane_base(self):
        return self._multi_plane_base

    @property
    def z_source(self):
        return self._z_source

    @property
    def z_source_convention(self):
        return self._z_source_convention

    @property
    def z_lens_convention(self):
        return self._z_lens_convention

    @property
    def T_ij_start(self):
        return self._T_ij_start

    @T_ij_start.setter
    def T_ij_start(self, T_ij_start):
        self._T_ij_start = T_ij_start

    @property
    def T_ij_stop(self):
        return self._T_ij_stop

    @T_ij_stop.setter
    def T_ij_stop(self, T_ij_stop):
        self._T_ij_stop = T_ij_stop

    def _set_source_distances(self, z_source):
        """Compute the relevant angular diameter distances to a specific source
        redshift.

        :param z_source: float, source redshift
        :return: self variables
        """
        self._z_source = z_source
        (
            self._T_ij_start,
            self._T_ij_stop,
        ) = self._multi_plane_base.transverse_distance_start_stop(
            z_start=0, z_stop=z_source, include_z_start=False
        )
        self._T_z_source = self._multi_plane_base._cosmo_bkg.T_xy(0, z_source)

    def observed2flat_convention(self, kwargs_lens):
        """

        :param kwargs_lens: keyword argument list of lens model parameters in the observed convention
        :return: kwargs_lens positions mapped into angular position without lensing along its LOS
        """
        return self._convention(kwargs_lens)

    def ray_shooting(
        self, theta_x, theta_y, kwargs_lens, check_convention=True, k=None
    ):
        """Ray-tracing (backwards light cone) to the default z_source redshift.

        :param theta_x: angle in x-direction on the image (usually arc seconds, in the
            same convention as lensing deflection angles)
        :param theta_y: angle in y-direction on the image (usually arc seconds, in the
            same convention as lensing deflection angles)
        :param kwargs_lens: lens model keyword argument list
        :param check_convention: flag to check the image position convention (leave this
            alone)
        :return: angles in the source plane
        """
        self._check_raise(k=k)
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        x = np.zeros_like(theta_x, dtype=float)
        y = np.zeros_like(theta_y, dtype=float)
        alpha_x = np.array(theta_x)
        alpha_y = np.array(theta_y)

        x, y, _, _ = self._multi_plane_base.ray_shooting_partial_comoving(
            x,
            y,
            alpha_x,
            alpha_y,
            z_start=0,
            z_stop=self._z_source,
            kwargs_lens=kwargs_lens,
            T_ij_start=self._T_ij_start,
            T_ij_end=self._T_ij_stop,
        )

        beta_x, beta_y = self.co_moving2angle_source(x, y)

        return beta_x, beta_y

    def ray_shooting_partial(
        self,
        theta_x,
        theta_y,
        alpha_x,
        alpha_y,
        z_start,
        z_stop,
        kwargs_lens,
        include_z_start=False,
        T_ij_start=None,
        T_ij_end=None,
        check_convention=True,
    ):
        """Ray-tracing through parts of the cone, starting with (x,y) in angular units
        as seen on the sky without lensing and angles (alpha_x, alpha_y) as seen at
        redshift z_start and then backwards to redshift z_stop.

        :param theta_x: angular position on the sky [arcsec]
        :param theta_y: angular position on the sky [arcsec]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param include_z_start: bool, if True, includes the computation of the
            deflection angle at the same redshift as the start of the ray-tracing.
            ATTENTION: deflection angles at the same redshift as z_stop will be computed
            always! This can lead to duplications in the computation of deflection
            angles.
        :param T_ij_start: transverse angular distance between the starting redshift to
            the first lens plane to follow. If not set, will compute the distance each
            time this function gets executed.
        :param T_ij_end: transverse angular distance between the last lens plane being
            computed and z_end. If not set, will compute the distance each time this
            function gets executed.
        :param check_convention: flag to check the image position convention (leave this
            alone)
        :return: angular position and angles at redshift z_stop
        """
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.ray_shooting_partial(
            theta_x,
            theta_y,
            alpha_x,
            alpha_y,
            z_start,
            z_stop,
            kwargs_lens,
            include_z_start=include_z_start,
            T_ij_start=T_ij_start,
            T_ij_end=T_ij_end,
        )

    def ray_shooting_partial_comoving(
        self,
        x,
        y,
        alpha_x,
        alpha_y,
        z_start,
        z_stop,
        kwargs_lens,
        include_z_start=False,
        check_convention=True,
        T_ij_start=None,
        T_ij_end=None,
    ):
        """Ray-tracing through parts of the cone, starting with (x,y) co-moving
        distances and angles (alpha_x, alpha_y) at redshift z_start and then backwards
        to redshift z_stop.

        :param x: co-moving position [Mpc] / angle definition
        :param y: co-moving position [Mpc] / angle definition
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param include_z_start: bool, if True, includes the computation of the
            deflection angle at the same redshift as the start of the ray-tracing.
            ATTENTION: deflection angles at the same redshift as z_stop will be
            computed! This can lead to duplications in the computation of deflection
            angles.
        :param check_convention: flag to check the image position convention (leave this
            alone)
        :param T_ij_start: transverse angular distance between the starting redshift to
            the first lens plane to follow. If not set, will compute the distance each
            time this function gets executed.
        :param T_ij_end: transverse angular distance between the last lens plane being
            computed and z_end. If not set, will compute the distance each time this
            function gets executed.
        :return: co-moving position (modulo angle definition) and angles at redshift
            z_stop
        """

        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)

        return self._multi_plane_base.ray_shooting_partial_comoving(
            x,
            y,
            alpha_x,
            alpha_y,
            z_start,
            z_stop,
            kwargs_lens,
            include_z_start=include_z_start,
            T_ij_start=T_ij_start,
            T_ij_end=T_ij_end,
        )

    def transverse_distance_start_stop(self, z_start, z_stop, include_z_start=False):
        """Computes the transverse distance (T_ij) that is required by the ray-tracing
        between the starting redshift and the first deflector afterwards and the last
        deflector before the end of the ray-tracing.

        :param z_start: redshift of the start of the ray-tracing
        :param z_stop: stop of ray-tracing
        :param include_z_start: bool, i
        :return: T_ij_start, T_ij_end
        """
        return self._multi_plane_base.transverse_distance_start_stop(
            z_start, z_stop, include_z_start
        )

    def arrival_time(self, theta_x, theta_y, kwargs_lens, check_convention=True):
        """Light travel time relative to a straight path through the coordinate (0,0)
        Negative sign means earlier arrival time.

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens: lens model keyword argument list
        :return: travel time in unit of days
        """
        dt_geo, dt_grav = self.geo_shapiro_delay(
            theta_x, theta_y, kwargs_lens, check_convention=check_convention
        )
        return dt_geo + dt_grav

    def geo_shapiro_delay(self, theta_x, theta_y, kwargs_lens, check_convention=True):
        """Geometric and Shapiro (gravitational) light travel time relative to a
        straight path through the coordinate (0,0) Negative sign means earlier arrival
        time.

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens: lens model keyword argument list
        :param check_convention: boolean, if True goes through the lens model list and
            checks whether the positional conventions are satisfied.
        :return: geometric delay, gravitational delay [days]
        """
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)
        return self._multi_plane_base.geo_shapiro_delay(
            theta_x,
            theta_y,
            kwargs_lens,
            z_stop=self._z_source,
            T_z_stop=self._T_z_source,
            T_ij_end=self._T_ij_stop,
        )

    def alpha(self, theta_x, theta_y, kwargs_lens, check_convention=True, k=None):
        """Reduced deflection angle.

        :param theta_x: angle in x-direction
        :param theta_y: angle in y-direction
        :param kwargs_lens: lens model kwargs
        :param check_convention: flag to check the image position convention (leave this
            alone)
        :return: deflection angles in x and y directions
        """
        self._check_raise(k=k)
        beta_x, beta_y = self.ray_shooting(
            theta_x, theta_y, kwargs_lens, check_convention=check_convention
        )

        alpha_x = theta_x - beta_x
        alpha_y = theta_y - beta_y

        return alpha_x, alpha_y

    def hessian(
        self,
        theta_x,
        theta_y,
        kwargs_lens,
        k=None,
        diff=0.00000001,
        check_convention=True,
    ):
        """Computes the hessian components f_xx, f_yy, f_xy from f_x and f_y with
        numerical differentiation.

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching
            the lens model classes
        :param diff: numerical differential step (float)
        :param check_convention: boolean, if True goes through the lens model list and
            checks whether the positional conventions are satisfied.
        :return: f_xx, f_xy, f_yx, f_yy
        """
        self._check_raise(k=k)
        if check_convention and not self.ignore_observed_positions:
            kwargs_lens = self._convention(kwargs_lens)

        alpha_ra, alpha_dec = self.alpha(
            theta_x, theta_y, kwargs_lens, check_convention=False
        )

        alpha_ra_dx, alpha_dec_dx = self.alpha(
            theta_x + diff, theta_y, kwargs_lens, check_convention=False
        )
        alpha_ra_dy, alpha_dec_dy = self.alpha(
            theta_x, theta_y + diff, kwargs_lens, check_convention=False
        )

        dalpha_rara = (alpha_ra_dx - alpha_ra) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra) / diff
        dalpha_decra = (alpha_dec_dx - alpha_dec) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def hessian_z1z2(self, z1, z2, theta_x, theta_y, kwargs_lens, diff=0.00000001):
        """Computes Hessian matrix when Observed at z1 with rays going to z2 with z1 <
        z2.

        :param z1: Observer redshift
        :param z2: source redshift
        :param theta_x: angular position and direction of the ray
        :param theta_y: angular position and direction of the ray
        :param kwargs_lens: list of keyword arguments of lens model parameters matching
            the lens model classes
        :param diff: numerical differential step (float)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        T_0z1 = self._multi_plane_base._cosmo_bkg.T_xy(0, z1)
        x = theta_x * T_0z1
        y = theta_x * T_0z1
        x_s0, y_s0, _, _ = self.ray_shooting_partial_comoving(
            x=x,
            y=y,
            alpha_x=theta_x,
            alpha_y=theta_y,
            z_start=z1,
            z_stop=z2,
            kwargs_lens=kwargs_lens,
            include_z_start=False,
            check_convention=True,
            T_ij_start=None,
            T_ij_end=None,
        )
        beta_x0, beta_y0 = self.co_moving2angle_z1_z2(x_s0, y_s0, z1=z1, z2=z2)
        alpha_ra = theta_x - beta_x0
        alpha_dec = theta_y - beta_y0

        x_s_dx, y_s_dx, _, _ = self.ray_shooting_partial_comoving(
            x=x,
            y=y,
            alpha_x=theta_x + diff,
            alpha_y=theta_y,
            z_start=z1,
            z_stop=z2,
            kwargs_lens=kwargs_lens,
            include_z_start=False,
            check_convention=True,
            T_ij_start=None,
            T_ij_end=None,
        )
        beta_x_dx, beta_y_dx = self.co_moving2angle_z1_z2(x_s_dx, y_s_dx, z1=z1, z2=z2)
        alpha_ra_dx = theta_x + diff - beta_x_dx
        alpha_dec_dx = theta_y - beta_y_dx

        x_s_dy, y_s_dy, _, _ = self.ray_shooting_partial_comoving(
            x=x,
            y=y,
            alpha_x=theta_x,
            alpha_y=theta_y + diff,
            z_start=z1,
            z_stop=z2,
            kwargs_lens=kwargs_lens,
            include_z_start=False,
            check_convention=True,
            T_ij_start=None,
            T_ij_end=None,
        )
        beta_x_dy, beta_y_dy = self.co_moving2angle_z1_z2(x_s_dy, y_s_dy, z1=z1, z2=z2)
        alpha_ra_dy = theta_x - beta_x_dy
        alpha_dec_dy = theta_y + diff - beta_y_dy

        dalpha_rara = (alpha_ra_dx - alpha_ra) / diff
        dalpha_radec = (alpha_ra_dy - alpha_ra) / diff
        dalpha_decra = (alpha_dec_dx - alpha_dec) / diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec) / diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def co_moving2angle_z1_z2(self, x, y, z1, z2):
        """Computes angle for co-moving distance at z=z2 when seen from z=z1.

        :param x: co-moving distance at z=z2
        :param y: co-moving distance at z=z2
        :param z1: redshift of observer
        :param z2: redshift of source
        :return: theta_z, theta_y
        """
        T_z1_z2 = self._multi_plane_base._cosmo_bkg.T_xy(z1, z2)
        theta_x = x / T_z1_z2
        theta_y = y / T_z1_z2
        return theta_x, theta_y

    def co_moving2angle_source(self, x, y):
        """Special case of the co_moving2angle definition at the source redshift.

        :param x: co-moving distance
        :param y: co-moving distance
        :return: angles on the sky at the nominal source plane
        """
        T_z = self._T_z_source
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    def set_static(self, kwargs):
        """

        :param kwargs: lens model keyword argument list
        :return: lens model keyword argument list with positional parameters all in flat sky coordinates
        """

        kwargs = self.observed2flat_convention(kwargs)
        self.ignore_observed_positions = True
        return self._multi_plane_base.set_static(kwargs)

    def set_dynamic(self):
        """

        :return:
        """
        self.ignore_observed_positions = False
        self._multi_plane_base.set_dynamic()

    @staticmethod
    def _check_raise(k=None):
        """Checks whether no option to select a specific subset of deflector models is
        selected, as this feature is not yet supported in multi-plane.

        :param k: parameter that optionally indicates a sub-set of lens models being
            executed for single plane
        :return: None, optional raise
        """
        if k is not None:
            raise ValueError(
                "no specific selection of a subset of lens models supported in multi-plane mode. Please"
                "use single plane mode or generate new instance of LensModel of the subset of profiles."
            )


@export
class PhysicalLocation(object):
    """center_x and center_y kwargs correspond to angular location of deflectors without
    lensing along the LOS."""

    def __call__(self, kwargs_lens):
        return kwargs_lens


@export
class LensedLocation(object):
    """center_x and center_y kwargs correspond to observed (lensed) locations of
    deflectors given a model for the line of sight structure, compute the angular
    position of the deflector without lensing contribution along the LOS."""

    def __init__(self, multiplane_instance, observed_convention_index):
        """

        :param multiplane_instance: instance of the MultiPlane class
        :param observed_convention_index: list of lens model indexes to be modelled in the observed plane
        """

        self._multiplane = multiplane_instance

        if len(observed_convention_index) == 1:
            self._inds = observed_convention_index
        else:
            inds = np.array(observed_convention_index)
            z = []

            for ind in inds:
                z.append(multiplane_instance._lens_redshift_list[ind])

            sort = np.argsort(z)

            self._inds = inds[sort]

    def __call__(self, kwargs_lens):
        new_kwargs = deepcopy(kwargs_lens)

        for ind in self._inds:
            theta_x = kwargs_lens[ind]["center_x"]
            theta_y = kwargs_lens[ind]["center_y"]
            zstop = self._multiplane._lens_redshift_list[ind]

            x, y, _, _ = self._multiplane.ray_shooting_partial_comoving(
                0,
                0,
                theta_x,
                theta_y,
                0,
                zstop,
                new_kwargs,
                T_ij_start=None,
                T_ij_end=None,
            )

            T = self._multiplane.T_z_list[ind]
            new_kwargs[ind]["center_x"] = x / T
            new_kwargs[ind]["center_y"] = y / T

        return new_kwargs
