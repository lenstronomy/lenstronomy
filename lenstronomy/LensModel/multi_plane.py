import numpy as np
from lenstronomy.Cosmo.background import Background
from lenstronomy.LensModel.single_plane import SinglePlane
import lenstronomy.Util.constants as const


class MultiPlane(object):
    """
    Multi-plane lensing class
    """

    def __init__(self, z_source, lens_model_list, redshift_list, cosmo=None, **lensmodel_kwargs):
        """

        :param cosmo: instance of astropy.cosmology
        :return: Background class with instance of astropy.cosmology
        """
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self._cosmo_bkg = Background(cosmo)
        self._z_source = z_source
        if not len(lens_model_list) == len(redshift_list):
            raise ValueError("The length of lens_model_list does not correspond to redshift_list")
        self._lens_model_list = lens_model_list
        self._redshift_list = redshift_list
        if len(lens_model_list) < 1:
            self._sorted_redshift_index = []
        else:
            self._sorted_redshift_index = self._index_ordering(redshift_list)
        self._lens_model = SinglePlane(lens_model_list, **lensmodel_kwargs)
        z_before = 0
        self._T_ij_list = []
        self._T_z_list = []
        self._reduced2physical_factor = []
        for idex in self._sorted_redshift_index:
            z_lens = self._redshift_list[idex]
            if z_before == z_lens:
                delta_T = 0
            else:
                delta_T = self._cosmo_bkg.T_xy(z_before, z_lens)
            self._T_ij_list.append(delta_T)
            T_z = self._cosmo_bkg.T_xy(0, z_lens)
            self._T_z_list.append(T_z)
            factor = self._cosmo_bkg.D_xy(0, z_source) / self._cosmo_bkg.D_xy(z_lens, z_source)
            self._reduced2physical_factor.append(factor)
            z_before = z_lens
        delta_T = self._cosmo_bkg.T_xy(z_before, z_source)
        self._T_ij_list.append(delta_T)
        self._T_z_source = self._cosmo_bkg.T_xy(0, z_source)
        sum_partial = np.sum(self._T_ij_list)
        if np.abs(sum_partial - self._T_z_source) > 0.1:
            print("Numerics in multi-plane compromised by too narrow spacing of too many redshift bins")

    def ray_shooting(self, theta_x, theta_y, kwargs_lens, k=None):
        """
        ray-tracing (backwards light cone)

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :return: angles in the source plane
        """
        x = np.zeros_like(theta_x)
        y = np.zeros_like(theta_y)
        alpha_x = theta_x
        alpha_y = theta_y
        i = -1
        for i, idex in enumerate(self._sorted_redshift_index):
            delta_T = self._T_ij_list[i]
            x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
            alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)
        delta_T = self._T_ij_list[i+1]
        x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
        beta_x, beta_y = self._co_moving2angle_source(x, y)
        return beta_x, beta_y

    def ray_shooting_partial(self, x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens, keep_range=False,
                             include_z_start=False):
        """
        ray-tracing through parts of the coin, starting with (x,y) and angles (alpha_x, alpha_y) at redshift z_start
        and then backwards to redshfit z_stop

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param keep_range: bool, if True, only computes the angular diameter ratio between the first and last step once
        :return: co-moving position and angles at redshift z_stop
        """
        z_lens_last = z_start
        first_deflector = True
        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._redshift_list[idex]
            if self._start_condition(include_z_start,z_lens,z_start) and z_lens <= z_stop:
            #if z_lens > z_start and z_lens <= z_stop:
                if first_deflector is True:
                    if keep_range is True:
                        if not hasattr(self, '_cosmo_bkg_T_start'):
                            self._cosmo_bkg_T_start = self._cosmo_bkg.T_xy(z_start, z_lens)
                        delta_T = self._cosmo_bkg_T_start
                    else:
                        delta_T = self._cosmo_bkg.T_xy(z_start, z_lens)
                    first_deflector = False
                else:
                    delta_T = self._T_ij_list[i]
                x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
                alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)
                z_lens_last = z_lens
        if keep_range is True:
            if not hasattr(self, '_cosmo_bkg_T_stop'):
                self._cosmo_bkg_T_stop = self._cosmo_bkg.T_xy(z_lens_last, z_stop)
            delta_T = self._cosmo_bkg_T_stop
        else:
            delta_T = self._cosmo_bkg.T_xy(z_lens_last, z_stop)

        x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
        return x, y, alpha_x, alpha_y

    def ray_shooting_partial_steps(self, x, y, alpha_x, alpha_y, z_start, z_stop, kwargs_lens,
                             include_z_start=False):
        """
        ray-tracing through parts of the coin, starting with (x,y) and angles (alpha_x, alpha_y) at redshift z_start
        and then backwards to redshfit z_stop.

        This function differs from 'ray_shooting_partial' in that it returns the angular position of the ray
        at each lens plane.

        :param x: co-moving position [Mpc]
        :param y: co-moving position [Mpc]
        :param alpha_x: ray angle at z_start [arcsec]
        :param alpha_y: ray angle at z_start [arcsec]
        :param z_start: redshift of start of computation
        :param z_stop: redshift where output is computed
        :param kwargs_lens: lens model keyword argument list
        :param keep_range: bool, if True, only computes the angular diameter ratio between the first and last step once
        :return: co-moving position and angles at redshift z_stop
        """
        z_lens_last = z_start
        first_deflector = True

        pos_x, pos_y, redshifts, Tz_list = [], [], [], []
        pos_x.append(x)
        pos_y.append(y)
        redshifts.append(z_start)
        Tz_list.append(self._cosmo_bkg.T_xy(0, z_start))

        current_z = z_lens_last

        for i, idex in enumerate(self._sorted_redshift_index):

            z_lens = self._redshift_list[idex]

            if self._start_condition(include_z_start,z_lens,z_start) and z_lens <= z_stop:

                if z_lens != current_z:
                    new_plane = True
                    current_z = z_lens

                else:
                    new_plane = False

                if first_deflector is True:
                    delta_T = self._cosmo_bkg.T_xy(z_start, z_lens)

                    first_deflector = False
                else:
                    delta_T = self._T_ij_list[i]
                x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
                alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)
                z_lens_last = z_lens

                if new_plane:

                    pos_x.append(x)
                    pos_y.append(y)
                    redshifts.append(z_lens)
                    Tz_list.append(self._T_z_list[i])

        delta_T = self._cosmo_bkg.T_xy(z_lens_last, z_stop)

        x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)

        pos_x.append(x)
        pos_y.append(y)
        redshifts.append(self._z_source)
        Tz_list.append(self._T_z_source)

        return pos_x, pos_y, redshifts, Tz_list

    def arrival_time(self, theta_x, theta_y, kwargs_lens, k=None):
        """
        light travel time relative to a straight path through the coordinate (0,0)
        Negative sign means earlier arrival time

        :param theta_x: angle in x-direction on the image
        :param theta_y: angle in y-direction on the image
        :param kwargs_lens:
        :return: travel time in unit of days
        """
        dt_grav = np.zeros_like(theta_x)
        dt_geo = np.zeros_like(theta_x)
        x = np.zeros_like(theta_x)
        y = np.zeros_like(theta_y)
        alpha_x = theta_x
        alpha_y = theta_y
        i = 0
        for i, idex in enumerate(self._sorted_redshift_index):
            z_lens = self._redshift_list[idex]
            delta_T = self._T_ij_list[i]
            dt_geo_new = self._geometrical_delay(alpha_x, alpha_y, delta_T)
            x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
            dt_grav_new = self._gravitational_delay(x, y, kwargs_lens, i, z_lens)
            alpha_x, alpha_y = self._add_deflection(x, y, alpha_x, alpha_y, kwargs_lens, i)
            dt_geo = dt_geo + dt_geo_new
            dt_grav = dt_grav + dt_grav_new
        delta_T = self._T_ij_list[i + 1]
        dt_geo += self._geometrical_delay(alpha_x, alpha_y, delta_T)
        x, y = self._ray_step(x, y, alpha_x, alpha_y, delta_T)
        beta_x, beta_y = self._co_moving2angle_source(x, y)
        dt_geo -= self._geometrical_delay(beta_x, beta_y, self._T_z_source)
        return dt_grav + dt_geo

    def alpha(self, theta_x, theta_y, kwargs_lens, k=None):
        """
        reduced deflection angle

        :param theta_x: angle in x-direction
        :param theta_y: angle in y-direction
        :param kwargs_lens: lens model kwargs
        :return:
        """
        beta_x, beta_y = self.ray_shooting(theta_x, theta_y, kwargs_lens)
        alpha_x = theta_x - beta_x
        alpha_y = theta_y - beta_y
        return alpha_x, alpha_y

    def hessian(self, theta_x, theta_y, kwargs_lens, k=None, diff=0.00000001):
        """
        computes the hessian components f_xx, f_yy, f_xy from f_x and f_y with numerical differentiation

        :param theta_x: x-position (preferentially arcsec)
        :type theta_x: numpy array
        :param theta_y: y-position (preferentially arcsec)
        :type theta_y: numpy array
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential step (float)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        alpha_ra, alpha_dec = self.alpha(theta_x, theta_y, kwargs_lens)

        alpha_ra_dx, alpha_dec_dx = self.alpha(theta_x + diff, theta_y, kwargs_lens)
        alpha_ra_dy, alpha_dec_dy = self.alpha(theta_x, theta_y + diff, kwargs_lens)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def _index_ordering(self, redshift_list):
        """

        :param redshift_list: list of redshifts
        :return: indexes in acending order to be evaluated (from z=0 to z=z_source)
        """
        redshift_list = np.array(redshift_list)
        sort_index = np.argsort(redshift_list[redshift_list < self._z_source])
        if len(sort_index) < 1:
            raise ValueError("There is no lens object between observer at z=0 and source at z=%s" % self._z_source)
        return sort_index

    def _reduced2physical_deflection(self, alpha_reduced, idex_lens):
        """
        alpha_reduced = D_ds/Ds alpha_physical

        :param alpha_reduced: reduced deflection angle
        :param z_lens: lens redshift
        :param z_source: source redshift
        :return: physical deflection angle
        """
        factor = self._reduced2physical_factor[idex_lens]
        #factor = self._cosmo_bkg.D_xy(0, z_source) / self._cosmo_bkg.D_xy(z_lens, z_source)
        return alpha_reduced * factor

    def _gravitational_delay(self, x, y, kwargs_lens, idex, z_lens):
        """

        :param x: co-moving coordinate at the lens plane
        :param y: co-moving coordinate at the lens plane
        :param kwargs_lens: lens model keyword arguments
        :param z_lens: redshift of the deflector
        :param idex: index of the lens model
        :return: gravitational delay in units of days as seen at z=0
        """
        theta_x, theta_y = self._co_moving2angle(x, y, idex)
        potential = self._lens_model.potential(theta_x, theta_y, kwargs_lens, k=self._sorted_redshift_index[idex])
        delay_days = self._lensing_potential2time_delay(potential, z_lens, z_source=self._z_source)
        return -delay_days

    def _geometrical_delay(self, alpha_x, alpha_y, delta_T):
        """
        geometrical delay (evaluated at z=0) of a light ray with an angle relative to the shortest path

        :param alpha_x: angle relative to a straight path
        :param alpha_y: angle relative to a straight path
        :param delta_T: transversal diameter distance between the start and end of the ray
        :return: geometrical delay in units of days
        """
        dt_days = (alpha_x**2 + alpha_y**2) / 2. * delta_T * const.Mpc / const.c / const.day_s * const.arcsec**2
        return dt_days

    def _lensing_potential2time_delay(self, potential, z_lens, z_source):
        """
        transforms the lensing potential (in units arcsec^2) to a gravitational time-delay as measured at z=0

        :param potential: lensing potential
        :param z_lens: redshift of the deflector
        :param z_source: redshift of source for the definition of the lensing quantities
        :return: gravitational time-delay in units of days
        """
        D_dt = self._cosmo_bkg.D_dt(z_lens, z_source)
        delay_days = const.delay_arcsec2days(potential, D_dt)
        return delay_days

    def _co_moving2angle(self, x, y, idex):
        """
        transforms co-moving distances Mpc into angles on the sky (radian)

        :param x: co-moving distance
        :param y: co-moving distance
        :param z_lens: redshift of plane
        :return: angles on the sky
        """
        T_z = self._T_z_list[idex]
        #T_z = self._cosmo_bkg.T_xy(0, z_lens)
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    def _co_moving2angle_source(self, x, y):
        """
        special case of the co_moving2angle definition at the source redshift

        :param x:
        :param y:
        :return:
        """
        T_z = self._T_z_source
        theta_x = x / T_z
        theta_y = y / T_z
        return theta_x, theta_y

    def _ray_step(self, x, y, alpha_x, alpha_y, delta_T):
        """
        ray propagation with small angle approximation

        :param x: co-moving x-position
        :param y: co-moving y-position
        :param alpha_x: deflection angle in x-direction at (x, y)
        :param alpha_y: deflection angle in y-direction at (x, y)
        :param delta_T: transversal angular diameter distance to the next step
        :return:
        """
        x_ = x + alpha_x * delta_T
        y_ = y + alpha_y * delta_T
        return x_, y_

    def _add_deflection(self, x, y, alpha_x, alpha_y, kwargs_lens, idex):
        """
        adds the pyhsical deflection angle of a single lens plane to the deflection field

        :param x: co-moving distance at the deflector plane
        :param y: co-moving distance at the deflector plane
        :param alpha_x: physical angle (radian) before the deflector plane
        :param alpha_y: physical angle (radian) before the deflector plane
        :param kwargs_lens: lens model parameter kwargs
        :param idex: index of the lens model to be added
        :param idex_lens: redshift of the deflector plane
        :return: updated physical deflection after deflector plane (in a backwards ray-tracing perspective)
        """
        theta_x, theta_y = self._co_moving2angle(x, y, idex)
        alpha_x_red, alpha_y_red = self._lens_model.alpha(theta_x, theta_y, kwargs_lens, k=self._sorted_redshift_index[idex])
        alpha_x_phys = self._reduced2physical_deflection(alpha_x_red, idex)
        alpha_y_phys = self._reduced2physical_deflection(alpha_y_red, idex)
        alpha_x_new = alpha_x - alpha_x_phys
        alpha_y_new = alpha_y - alpha_y_phys
        return alpha_x_new, alpha_y_new

    def _start_condition(self, inclusive, z_lens, z_start):

        if inclusive:
            return z_lens >= z_start
        else:
            return z_lens > z_start
