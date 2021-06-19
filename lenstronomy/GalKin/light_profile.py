import numpy as np
import copy
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from lenstronomy.LightModel.light_model import LightModel

__all__ = ['LightProfile']


class LightProfile(object):
    """
    class to deal with the light distribution for GalKin

    In particular, this class allows for:
     - (faster) interpolated calculation for a given profile (for a range that the Jeans equation is computed)
     - drawing 3d and 2d distributions from a given (spherical) profile
       (within bounds where the Jeans equation is expected to be accurate)
     - 2d projected profiles within the 3d integration range (truncated)

    """
    def __init__(self, profile_list, interpol_grid_num=2000, max_interpolate=1000, min_interpolate=0.001,
                 max_draw=None):
        """

        :param profile_list: list of light profiles for LightModel module (must support light_3d() functionalities)
        :param interpol_grid_num: int; number of interpolation steps (logarithmically between min and max value)
        :param max_interpolate: float; maximum interpolation of 3d light profile
        :param min_interpolate: float; minimum interpolate (and also drawing of light profile)
        :param max_draw: float; (optional) if set, draws up to this radius, else uses max_interpolate value
        """

        self.light_model = LightModel(light_model_list=profile_list)
        self._interp_grid_num = interpol_grid_num
        self._max_interpolate = max_interpolate
        self._min_interpolate = min_interpolate
        if max_draw is None:
            max_draw = max_interpolate
        self._max_draw = max_draw

    def light_3d(self, r, kwargs_list):
        """
        three-dimensional light profile

        :param r: 3d radius
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: flux per 3d volume at radius r
        """
        light_3d = self.light_model.light_3d(r, kwargs_list)
        return light_3d

    def light_3d_interp(self, r, kwargs_list, new_compute=False):
        """
        interpolated three-dimensional light profile within bounds [min_interpolate, max_interpolate]
        in logarithmic units with interpol_grid_num numbers of interpolation steps

        :param r: 3d radius
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :param new_compute: boolean, if True, re-computes the interpolation
         (becomes valid with updated kwargs_list argument)
        :return: flux per 3d volume at radius r
        """
        if not hasattr(self, '_f_light_3d') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            light_3d_array = self.light_model.light_3d(r_array, kwargs_list)
            light_3d_array[light_3d_array < 10 ** (-1000)] = 10 ** (-1000)
            f = interp1d(np.log(r_array), np.log(light_3d_array), fill_value=(np.log(light_3d_array[0]), -1000),
                         bounds_error=False)  # "extrapolate"
            self._f_light_3d = f
        return np.exp(self._f_light_3d(np.log(r)))

    def light_2d(self, R, kwargs_list):
        """
        projected light profile (integrated to infinity in the projected axis)

        :param R: projected 2d radius
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: projected surface brightness
        """
        kwargs_light_circularized = self._circularize_kwargs(kwargs_list)
        return self.light_model.surface_brightness(R, 0, kwargs_light_circularized)

    def _circularize_kwargs(self, kwargs_list):
        """

        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: circularized arguments
        """
        # TODO make sure averaging is done azimuthally
        if not hasattr(self, '_kwargs_light_circularized'):
            kwargs_list_copy = copy.deepcopy(kwargs_list)
            kwargs_list_new = []
            for kwargs in kwargs_list_copy:
                if 'e1' in kwargs:
                    kwargs['e1'] = 0
                if 'e2' in kwargs:
                    kwargs['e2'] = 0
                kwargs_list_new.append({k: v for k, v in kwargs.items() if not k in ['center_x', 'center_y']})
            self._kwargs_light_circularized = kwargs_list_new
        return self._kwargs_light_circularized

    def _light_2d_finite_single(self, R, kwargs_list):
        """
        projected light profile (integrated to FINITE 3d boundaries from the max_interpolate)
        for a single float number of R

        :param R: projected 2d radius (between min_interpolate and max_interpolate)
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: projected surface brightness
        """

        # here we perform a logarithmic integral
        stop = np.log10(np.maximum(np.sqrt(self._max_interpolate**2 - R**2), self._min_interpolate + 0.00001))
        x = np.logspace(start=np.log10(self._min_interpolate), stop=stop, num=self._interp_grid_num)
        r_array = np.sqrt(x**2 + R**2)
        flux_r = self.light_3d(r_array, kwargs_list)
        dlog_r = (np.log10(x[2]) - np.log10(x[1])) * np.log(10)
        flux_r *= dlog_r * x

        # linear integral
        #x = np.linspace(start=self._min_interpolate, stop=np.sqrt(self._max_interpolate ** 2 - R ** 2),
        #                num=self._interp_grid_num)
        #r_array = np.sqrt(x ** 2 + R ** 2)
        #dr = x[1] - x[0]
        #flux_r = self.light_3d(r_array + dr / 2, kwargs_circ)
        #dr = x[1] - x[0]
        #flux_r *= dr

        flux_R = np.sum(flux_r)
        # perform finite integral

        #out = integrate.quad(lambda x: self.light_3d(np.sqrt(R ** 2 + x ** 2), kwargs_circ), self._min_interpolate,
        #                     np.sqrt(self._max_interpolate**2 - R**2))
        #print(out_1, out, 'test')
        #flux_R = out[0]
        return flux_R * 2  # integral in both directions

    def light_2d_finite(self, R, kwargs_list):
        """
        projected light profile (integrated to FINITE 3d boundaries from the max_interpolate)

        :param R: projected 2d radius (between min_interpolate and max_interpolate
        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :return: projected surface brightness
        """

        kwargs_circ = self._circularize_kwargs(kwargs_list)
        n = len(np.atleast_1d(R))
        if n <= 1:
            return self._light_2d_finite_single(R, kwargs_circ)
        else:
            light_2d = np.zeros(n)
            for i, R_i in enumerate(R):
                light_2d[i] = self._light_2d_finite_single(R_i, kwargs_circ)
            return light_2d

    def draw_light_2d_linear(self, kwargs_list, n=1, new_compute=False):
        """
        constructs the CDF and draws from it random realizations of projected radii R
        The interpolation of the CDF is done in linear projected radius space

        :param kwargs_list: list of keyword arguments of light profiles (see LightModule)
        :param n: int; number of draws
        :param new_compute: boolean, if True, re-computes the interpolation
         (becomes valid with updated kwargs_list argument)
        :return: draw of projected radius for the given light profile distribution
        """
        if not hasattr(self, '_light_cdf') or new_compute is True:
            r_array = np.linspace(self._min_interpolate, self._max_draw, self._interp_grid_num)
            cum_sum = np.zeros_like(r_array)
            sum = 0
            for i, r in enumerate(r_array):
                if i == 0:
                    cum_sum[i] = 0
                else:
                    sum += self.light_2d(r, kwargs_list) * r
                    cum_sum[i] = copy.deepcopy(sum)
            cum_sum_norm = cum_sum/cum_sum[-1]
            f = interp1d(cum_sum_norm, r_array)
            self._light_cdf = f
        cdf_draw = np.random.uniform(0., 1, n)
        r_draw = self._light_cdf(cdf_draw)
        return r_draw

    def draw_light_2d(self, kwargs_list, n=1, new_compute=False):
        """
        constructs the CDF and draws from it random realizations of projected radii R
        CDF is constructed in logarithmic projected radius spacing

        :param kwargs_list: light model keyword argument list
        :param n: int, number of draws per functino call
        :param new_compute: re-computes the interpolated CDF
        :return: realization of projected radius following the distribution of the light model
        """
        if not hasattr(self, '_light_cdf_log') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_draw), self._interp_grid_num)
            cum_sum = np.zeros_like(r_array)
            sum = 0
            for i, r in enumerate(r_array):
                if i == 0:
                    cum_sum[i] = 0
                else:
                    sum += self.light_2d(r, kwargs_list) * r * r
                    cum_sum[i] = copy.deepcopy(sum)
            cum_sum_norm = cum_sum/cum_sum[-1]
            f = interp1d(cum_sum_norm, np.log(r_array))
            self._light_cdf_log = f
        cdf_draw = np.random.uniform(0., 1, n)
        r_log_draw = self._light_cdf_log(cdf_draw)
        return np.exp(r_log_draw)

    def draw_light_3d(self, kwargs_list, n=1, new_compute=False):
        """
        constructs the CDF and draws from it random realizations of 3D radii r

        :param kwargs_list: light model keyword argument list
        :param n: int, number of draws per function call
        :param new_compute: re-computes the interpolated CDF
        :return: realization of projected radius following the distribution of the light model
        """
        if not hasattr(self, '_light_3d_cdf_log') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_draw), self._interp_grid_num)
            dlog_r = np.log10(r_array[1]) - np.log10(r_array[0])
            r_array_int = np.logspace(np.log10(self._min_interpolate) + dlog_r / 2, np.log10(self._max_draw) + dlog_r / 2, self._interp_grid_num)
            cum_sum = np.zeros_like(r_array)
            sum = 0
            for i, r in enumerate(r_array_int[:-1]):
                #if i == 0:
                #    cum_sum[i] = 0
                #else:
                    sum += self.light_3d(r, kwargs_list) * r**2 * (r_array[i+1] - r_array[i])# * r
                    cum_sum[i+1] = copy.deepcopy(sum)
            cum_sum_norm = cum_sum/cum_sum[-1]
            f = interp1d(cum_sum_norm, np.log(r_array))
            self._light_3d_cdf_log = f
        cdf_draw = np.random.uniform(0., 1, n)
        r_log_draw = self._light_3d_cdf_log(cdf_draw)
        return np.exp(r_log_draw)

    def delete_cache(self):
        """
        deletes cached interpolation function of the CDF for a specific light profile

        :return: None
        """
        if hasattr(self, '_light_cdf_log'):
            del self._light_cdf_log
        if hasattr(self, '_light_cdf'):
            del self._light_cdf
        if hasattr(self, '_f_light_3d'):
            del self._f_light_3d
        if hasattr(self, '_kwargs_light_circularized'):
            del self._kwargs_light_circularized
