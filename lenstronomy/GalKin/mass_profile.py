import math
import numpy as np
from scipy.interpolate import interp1d
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.constants as const
from lenstronomy.GalKin.cosmo import Cosmo


class MassProfile(object):
    """
    mass profile class
    """
    def __init__(self, profile_list, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}, kwargs_numerics={}):
        """

        :param profile_list:
        """
        kwargs_options = {'lens_model_list': profile_list}
        self.model = LensModel(profile_list)
        self.cosmo = Cosmo(kwargs_cosmo)
        self._interp_grid_num = kwargs_numerics.get('interpol_grid_num', 1000)
        self._max_interpolate = kwargs_numerics.get('max_interpolate', 100)
        self._min_interpolate = kwargs_numerics.get('min_interpolate', 0.0001)

    def mass_3d_interp(self, r, kwargs, new_compute=False):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        if not hasattr(self, '_log_mass_3d') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            mass_3d_array = self.model.mass_3d(r_array, kwargs)
            mass_3d_array[mass_3d_array < 10. ** (-10)] = 10. ** (-10)
            mass_dim_array = mass_3d_array * const.arcsec ** 3 * self.cosmo.D_d ** 2 * self.cosmo.D_s \
                       / self.cosmo.D_ds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
            f = interp1d(np.log(r_array), np.log(mass_dim_array/r_array), fill_value="extrapolate")
            self._log_mass_3d = f
        return np.exp(self._log_mass_3d(np.log(r))) * r

    def mass_3d(self, r, kwargs):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        mass_dimless = self.model.mass_3d(r, kwargs)
        mass_dim = mass_dimless * const.arcsec ** 3 * self.cosmo.D_d ** 2 * self.cosmo.D_s \
                       / self.cosmo.D_ds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
        return mass_dim


class MassProfile_old(object):
    """
    class to deal with arbitrary mass profiles
    """

    def __init__(self, kwargs):
        self._profile_type = kwargs['type']
        self._kwargs = kwargs

    def m_r(self, r):
        """
        returns mass enclosed < r of the mass profile
        :param r:
        :return:
        """
        if self._profile_type == 'power_law':
            return self.m_r_power_law(r)
        else:
            return 0

    def m_r_power_law(self, r):
        """

        :param r:
        :return:
        """
        theta_E = self._kwargs['theta_E']
        gamma = self._kwargs['gamma']
        Sigma_crit = 1
        D_d = 1
        rho_0 = Sigma_crit * theta_E**(gamma-1) * D_d**(gamma-1) * math.gamma(gamma/2.) / (np.sqrt(np.pi) * math.gamma((gamma-3)/2.))
        return rho_0 / r**gamma