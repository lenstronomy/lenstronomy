import numpy as np
import copy
from scipy.interpolate import interp1d
from lenstronomy.LightModel.light_model import LightModel

__all__ = ['LightProfile']


class LightProfile(object):
    """
    class to deal with the light distribution
    """
    def __init__(self, profile_list, interpol_grid_num=2000, max_interpolate=1000, min_interpolate=0.001):
        """

        :param profile_list:
        """
        self.light_model = LightModel(light_model_list=profile_list)
        self._interp_grid_num = interpol_grid_num
        self._max_interpolate = max_interpolate
        self._min_interpolate = min_interpolate

    def light_3d(self, r, kwargs_list):
        """

        :param kwargs_list:
        :return:
        """
        light_3d = self.light_model.light_3d(r, kwargs_list)
        return light_3d

    def light_3d_interp(self, r, kwargs_list, new_compute=False):
        """

        :param kwargs_list:
        :return:
        """
        if not hasattr(self, '_f_light_3d') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            light_3d_array = self.light_model.light_3d(r_array, kwargs_list)
            light_3d_array[light_3d_array < 10 ** (-100)] = 10 ** (-100)
            f = interp1d(np.log(r_array), np.log(light_3d_array), fill_value=(np.log(light_3d_array[0]), -1000),
                         bounds_error=False)  # "extrapolate"
            self._f_light_3d = f
        return np.exp(self._f_light_3d(np.log(r)))

    def light_2d(self, R, kwargs_list):
        """

        :param R:
        :param kwargs_list:
        :return:
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
        return self.light_model.surface_brightness(R, 0, self._kwargs_light_circularized)

    def draw_light_2d_linear(self, kwargs_list, n=1, new_compute=False, r_eff=1.):
        """
        constructs the CDF and draws from it random realizations of projected radii R
        :param kwargs_list:
        :return:
        """
        if not hasattr(self, '_light_cdf') or new_compute is True:
            r_array = np.linspace(self._min_interpolate, self._max_interpolate, self._interp_grid_num)
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
        :param kwargs_list: light model keyword argument list
        :param n: int, number of draws per functino call
        :param new_compute: re-computes the interpolated CDF
        :return: realization of projected radius following the distribution of the light model
        """
        if not hasattr(self, '_light_cdf_log') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
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
        :param n: int, number of draws per functino call
        :param new_compute: re-computes the interpolated CDF
        :return: realization of projected radius following the distribution of the light model
        """
        if not hasattr(self, '_light_3d_cdf_log') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            cum_sum = np.zeros_like(r_array)
            sum = 0
            for i, r in enumerate(r_array):
                if i == 0:
                    cum_sum[i] = 0
                else:
                    sum += self.light_3d(r, kwargs_list) * r * r**2  # 1x r for the log spacing and r**2 for the shell area
                    cum_sum[i] = copy.deepcopy(sum)
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
