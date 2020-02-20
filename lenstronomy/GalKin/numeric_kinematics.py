import numpy as np
from scipy.interpolate import interp1d

import lenstronomy.Util.constants as const
from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.LensModel.single_plane import SinglePlane


class NumericKinematics(Anisotropy):

    def __init__(self, kwargs_model, kwargs_cosmo, interpol_grid_num=500, log_integration=False, max_integrate=10,
                 min_integrate=0.001):
        """

        :param interpol_grid_num:
        :param log_integration:
        :param max_integrate:
        :param min_integrate:
        """
        mass_profile_list = kwargs_model.get('mass_profile_list')
        light_profile_list = kwargs_model.get('light_profile_list')
        anisotropy_model = kwargs_model.get('anisotropy_model')
        self._interp_grid_num = interpol_grid_num
        self._log_int = log_integration
        self._max_integrate = max_integrate  # maximal integration (and interpolation) in units of arcsecs
        self._min_integrate = min_integrate  # min integration (and interpolation) in units of arcsecs
        self._max_interpolate = max_integrate  # we chose to set the interpolation range to the integration range
        self._min_interpolate = min_integrate  # we chose to set the interpolation range to the integration range
        self.lightProfile = LightProfile(light_profile_list, interpol_grid_num=interpol_grid_num,
                                         max_interpolate=max_integrate, min_interpolate=min_integrate)
        Anisotropy.__init__(self, anisotropy_type=anisotropy_model)
        self.cosmo = Cosmo(**kwargs_cosmo)
        self._mass_profile = SinglePlane(mass_profile_list)

    def _sigma2_R(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified projected radius

        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return:
        """
        I_R_sigma2 = self._I_R_simga2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R = self.lightProfile.light_2d(R, kwargs_light)
        return I_R_sigma2 / I_R

    def _I_R_simga2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        equation A15 in Mamon&Lokas 2005 as a logarithmic numerical integral (if option is chosen)

        :param R: 2d projected radius (in angular units)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integral of A15 in Mamon&Lokas 2005
        """
        R = max(R, self._min_integrate)
        if self._log_int is True:
            min_log = np.log10(R+0.001)
            max_log = np.log10(self._max_integrate)
            r_array = np.logspace(min_log, max_log, self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dlog_r * r_array
        else:
            r_array = np.linspace(R+0.001, self._max_integrate, self._interp_grid_num)
            dr = r_array[2] - r_array[1]
            IR_sigma2_dr = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy) * dr
        IR_sigma2 = np.sum(IR_sigma2_dr) # integral from angle to physical scales
        return IR_sigma2 * 2 * const.G / (const.arcsec * self.cosmo.dd * const.Mpc)

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space) in Mamon&Lokas 2005

        :param r: 3d radius in arc seconds
        :param R: 2d projected radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return:
        """
        k_r = self.K(r, R, **kwargs_anisotropy)
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        m_r = self._mass_3d_interp(r, kwargs_mass)
        out = k_r * l_r * m_r / r
        return out

    def _mass_3d_interp(self, r, kwargs, new_compute=False):
        """

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :param new_compute: bool, if True, recomputes the interpolation
        :return: mass enclosed physical radius in kg
        """
        if not hasattr(self, '_log_mass_3d') or new_compute is True:
            r_array = np.logspace(np.log10(self._min_interpolate), np.log10(self._max_interpolate), self._interp_grid_num)
            mass_3d_array = self.mass_3d(r_array, kwargs)
            mass_3d_array[mass_3d_array < 10. ** (-10)] = 10. ** (-10)
            #mass_dim_array = mass_3d_array * const.arcsec ** 2 * self.cosmo.dd * self.cosmo.ds \
            #                 / self.cosmo.dds * const.Mpc * const.c ** 2 / (4 * np.pi * const.G)
            f = interp1d(np.log(r_array), np.log(mass_3d_array/r_array), fill_value="extrapolate")
            self._log_mass_3d = f
        return np.exp(self._log_mass_3d(np.log(r))) * r

    def mass_3d(self, r, kwargs):
        """
        mass enclosed a 3d radius

        :param r: in arc seconds
        :param kwargs: lens model parameters in arc seconds
        :return: mass enclosed physical radius in kg
        """
        mass_dimless = self._mass_profile.mass_3d(r, kwargs)
        mass_dim = mass_dimless * const.arcsec ** 2 * self.cosmo.dd * self.cosmo.ds / self.cosmo.dds * const.Mpc * \
                   const.c ** 2 / (4 * np.pi * const.G)
        return mass_dim
