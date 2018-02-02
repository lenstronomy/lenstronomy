from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.mass_profile import MassProfile
from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin.anisotropy import MamonLokasAnisotropy
from lenstronomy.GalKin.cosmo import Cosmo
import lenstronomy.GalKin.velocity_util as util
import lenstronomy.Util.constants as const

import numpy as np


class Galkin(object):
    """
    major class to compute velocity dispersion measurements given light and mass models
    """
    def __init__(self, mass_profile_list, light_profile_list, aperture_type='slit', anisotropy_model='isotropic',
                 fwhm=0.7, kwargs_numerics={}, kwargs_cosmo={'D_d': 1000, 'D_s': 2000, 'D_ds': 500}):
        self.massProfile = MassProfile(mass_profile_list, kwargs_cosmo, kwargs_numerics=kwargs_numerics)
        self.lightProfile = LightProfile(light_profile_list, kwargs_numerics=kwargs_numerics)
        self.aperture = Aperture(aperture_type)
        self.anisotropy = MamonLokasAnisotropy(anisotropy_model)
        self.FWHM = fwhm
        self.cosmo = Cosmo(kwargs_cosmo)
        #kwargs_numerics = {'sampling_number': 10000, 'interpol_grid_num': 5000, 'log_integration': False,
        #                   'max_integrate': 500}
        self._num_sampling = kwargs_numerics.get('sampling_number', 1000)
        self._interp_grid_num = kwargs_numerics.get('interpol_grid_num', 500)
        self._log_int = kwargs_numerics.get('log_integration', False)
        self._max_integrate = kwargs_numerics.get('max_integrate', 10)  # maximal integration (and interpolation) in units of arcsecs
        self._min_integrate = kwargs_numerics.get('min_integrate', 0.001)  # min integration (and interpolation) in units of arcsecs


    def vel_disp(self, kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_apertur, r_eff=1.):
        """
        computes the averaged LOS velocity dispersion in the slit (convolved)
        :param gamma:
        :param phi_E:
        :param r_eff:
        :param r_ani:
        :param R_slit:
        :param FWHM:
        :return:
        """
        sigma2_R_sum = 0
        for i in range(0, self._num_sampling):
            sigma2_R = self.draw_one_sigma2(kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_apertur, r_eff=r_eff)
            sigma2_R_sum += sigma2_R
        sigma_s2_average = sigma2_R_sum / self._num_sampling
        # apply unit conversion from arc seconds and deflections to physical velocity disperison in (km/s)
        sigma_s2_average *= 2 * const.G  # correcting for integral prefactor
        return np.sqrt(sigma_s2_average/(const.arcsec**2 * self.cosmo.D_d**2 * const.Mpc))/1000.  # in units of km/s

    def draw_one_sigma2(self, kwargs_mass, kwargs_light, kwargs_anisotropy, kwargs_aperture, r_eff=1.):
        """

        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :param kwargs_aperture:
        :return:
        """
        while True:
            R = self.lightProfile.draw_light_2d(kwargs_light, r_eff=r_eff)  # draw r
            x, y = util.draw_xy(R)  # draw projected R
            x_, y_ = util.displace_PSF(x, y, self.FWHM)  # displace via PSF
            bool = self.aperture.aperture_select(x_, y_, kwargs_aperture)
            if bool is True:
                break
        sigma2_R = self.sigma2_R(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        return sigma2_R

    def sigma2_R(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion
        :param R:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        I_R_sigma2 = self.I_R_simga2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R = self.lightProfile.light_2d(R, kwargs_light)
        #I_R = self.lightProfile._integrand_light(R, kwargs_light)
        return I_R_sigma2 / I_R

    def I_R_simga2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        equation A15 in Mamon&Lokas 2005 as a logarithmic numerical integral
        modulo pre-factor 2*G
        :param R:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
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
        IR_sigma2 = np.sum(IR_sigma2_dr)
        return IR_sigma2

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space)
        :param r:
        :param kwargs_mass:
        :param kwargs_light:
        :param kwargs_anisotropy:
        :return:
        """
        k_r = self.anisotropy.K(r, R, kwargs_anisotropy)
        l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        m_r = self.massProfile.mass_3d_interp(r, kwargs_mass)
        out = k_r * l_r * m_r / r
        return out
