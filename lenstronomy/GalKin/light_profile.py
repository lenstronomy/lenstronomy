import numpy as np
import copy
from scipy.interpolate import interp1d
from lenstronomy.LightModel.light_model import LightModel


class LightProfile(object):
    """
    class to deal with the light distribution
    """
    def __init__(self, profile_list=['HERNQUIST'], kwargs_numerics={}):
        """

        :param profile_list:
        """
        self.light_model = LightModel(light_model_list=profile_list, smoothing=0.000001)
        self._interp_grid_num = kwargs_numerics.get('interpol_grid_num', 1000)
        self._max_interpolate = kwargs_numerics.get('max_integrate', 100)
        self._min_interpolate = kwargs_numerics.get('min_integrate', 0.001)

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
            light_3d_array[light_3d_array < 10 **(-100)] = 10**(-100)
            f = interp1d(np.log(r_array), np.log(light_3d_array), fill_value="extrapolate")
            self._f_light_3d = f
        return np.exp(self._f_light_3d(np.log(r)))

    def light_2d(self, R, kwargs_list):
        """

        :param R:
        :param kwargs_list:
        :return:
        """
        kwargs_list_copy = copy.deepcopy(kwargs_list)
        kwargs_list_new = []
        for kwargs in kwargs_list_copy:
            if 'e1' in kwargs:
                kwargs['e1'] = 0
            if 'e2' in kwargs:
                kwargs['e2'] = 0
            kwargs_list_new.append({k: v for k, v in kwargs.items() if not k in ['center_x', 'center_y']})
        return self.light_model.surface_brightness(R, 0, kwargs_list_new)

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
        :param kwargs_list:
        :return:
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


class LightProfile_old(object):
    """
    class to deal with the light distribution
    """
    def __init__(self, profile_type='Hernquist'):
        self._profile_type = profile_type

    def draw_light(self, kwargs_light):
        """

        :param kwargs_light:
        :return:
        """
        if self._profile_type == 'Hernquist':
            r = self.P_r_hernquist(kwargs_light)
        else:
            raise ValueError('light profile %s not supported!')
        return r

    def P_r_hernquist(self, kwargs_light):
        """

        :param a: 0.551*r_eff
        :return: realisation of radius of Hernquist luminosity weighting in 3d
        """
        r_eff = kwargs_light['r_eff']
        a = 0.551 * r_eff
        P = np.random.uniform()  # draws uniform between [0,1)
        r = a*np.sqrt(P)*(np.sqrt(P)+1)/(1-P)  # solves analytically to r from P(r)
        return r
