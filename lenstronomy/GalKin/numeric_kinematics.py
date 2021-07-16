import numpy as np
from scipy.interpolate import interp1d

import lenstronomy.Util.constants as const
from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.anisotropy import Anisotropy
from lenstronomy.GalKin.cosmo import Cosmo
from lenstronomy.LensModel.single_plane import SinglePlane
import lenstronomy.GalKin.velocity_util as util

__all__ = ['NumericKinematics']


class NumericKinematics(Anisotropy):

    def __init__(self, kwargs_model, kwargs_cosmo, interpol_grid_num=1000, log_integration=True, max_integrate=1000,
                 min_integrate=0.0001, max_light_draw=None, lum_weight_int_method=True):
        """
        What we need:
        - max projected R to have ACCURATE I_R_sigma values
        - make sure everything outside cancels out (or is not rendered)

        :param interpol_grid_num: number of interpolation bins for integrand and interpolated functions
        :param log_integration: bool, if True, performs the numerical integral in log space distance (adviced)
         (only applies for lum_weight_int_method=True)
        :param max_integrate: maximum radius (in arc seconds) of the Jeans equation integral
         (assumes zero tracer particles outside this radius)
        :param max_light_draw: float; (optional) if set, draws up to this radius, else uses max_interpolate value
        :param lum_weight_int_method: bool, luminosity weighted dispersion integral to calculate LOS projected Jean's
         solution. ATTENTION: currently less accurate than 3d solution
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
        if max_light_draw is None:
            max_light_draw = max_integrate  # make sure the actual solution for the kinematics is only computed way inside the integral
        self.lightProfile = LightProfile(light_profile_list, interpol_grid_num=interpol_grid_num,
                                         max_interpolate=max_integrate, min_interpolate=min_integrate,
                                         max_draw=max_light_draw)
        Anisotropy.__init__(self, anisotropy_type=anisotropy_model)
        self.cosmo = Cosmo(**kwargs_cosmo)
        self._mass_profile = SinglePlane(mass_profile_list)
        self._lum_weight_int_method = lum_weight_int_method

    def sigma_s2(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified 3d and projected radius
        (if lum_weight_int_method=True then the 3d radius is not required and the function directly performs the
        luminosity weighted integral in projection at R)

        :param r: 3d radius (not needed for this calculation)
        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
         We refer to the Anisotropy() class for details on the parameters.
        :return: weighted line-of-sight projected velocity dispersion at projected radius R with weights I
        """
        if self._lum_weight_int_method is True:
            return self.sigma_s2_project(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        else:
            return self.sigma_s2_r(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy), 1

    def sigma_s2_project(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns luminosity-weighted los velocity dispersion for a specified projected radius R and weight

        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: line-of-sight projected velocity dispersion at projected radius R
        """
        # nominator is numerically to a finite distance, so luminosity weighting might be off
        # this could lead to an under-prediction of the velocity dispersion
        # so we ask the function _I_R_sigma2() to also return the numerical l(r)
        #I_R_sigma2, I_R = self._I_R_sigma2_interp(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        I_R_sigma2, I_R = self._I_R_sigma2_interp(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        #I_R = self.lightProfile.light_2d(R, kwargs_light)
        return I_R_sigma2 / I_R, 1

    def sigma_s2_r(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        returns unweighted los velocity dispersion for a specified 3d radius r at projected radius R

        :param r: 3d radius (not needed for this calculation)
        :param R: 2d projected radius (in angular units of arcsec)
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: line-of-sight projected velocity dispersion at projected radius R from 3d radius r
        """
        beta = self.beta_r(r, **kwargs_anisotropy)
        return (1 - beta * R ** 2 / r ** 2) * self.sigma_r2(r, kwargs_mass, kwargs_light, kwargs_anisotropy)

    def sigma_r2(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        computes numerically the solution of the Jeans equation for a specific 3d radius
        E.g. Equation (A1) of Mamon & Lokas https://arxiv.org/pdf/astro-ph/0405491.pdf
        l(r) \sigma_r(r) ^ 2 =  1/f(r) \int_r^{\infty} f(s) l(s) G M(s) / s^2 ds
        where l(r) is the 3d light profile
        M(s) is the enclosed 3d mass
        f is the solution to
        d ln(f)/ d ln(r) = 2 beta(r)

        :param r: 3d radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: sigma_r**2
        """
        # l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        l_r = self.lightProfile.light_3d(r, kwargs_light)
        f_r = self.anisotropy_solution(r, **kwargs_anisotropy)
        return 1 / f_r / l_r * self._jeans_solution_integral(r, kwargs_mass, kwargs_light, kwargs_anisotropy) * const.G / (const.arcsec * self.cosmo.dd * const.Mpc)

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

    def grav_potential(self, r, kwargs_mass):
        """
        Gravitational potential in SI units

        :param r: radius (arc seconds)
        :param kwargs_mass:
        :return: gravitational potential
        """
        mass_dim = self.mass_3d(r, kwargs_mass)
        grav_pot = -const.G * mass_dim / (r * const.arcsec * self.cosmo.dd * const.Mpc)
        return grav_pot

    def draw_light(self, kwargs_light):
        """

        :param kwargs_light: keyword argument (list) of the light model
        :return: 3d radius (if possible), 2d projected radius, x-projected coordinate, y-projected coordinate
        """
        r = self.lightProfile.draw_light_3d(kwargs_light, n=1)[0]
        R, x, y = util.project2d_random(r)
        return r, R, x, y

    def delete_cache(self):
        """
        delete interpolation function for a specific mass and light profile as well as for a specific anisotropy model

        :return:
        """
        if hasattr(self, '_log_mass_3d'):
            del self._log_mass_3d
        if hasattr(self, '_interp_jeans_integral'):
            del self._interp_jeans_integral
        if hasattr(self, '_interp_I_R_sigma2'):
            del self._interp_I_R_sigma2
        self.lightProfile.delete_cache()
        self.delete_anisotropy_cache()

    def _I_R_sigma2(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
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
        max_integrate = self._max_integrate  # make sure the integration of the Jeans equation is performed further out than the interpolation
        #if False:
        #    # linear integral near R
        #    lin_max = min(2 * R_, self._max_interpolate)
        #    lin_max = min(lin_max, R_+1)
        #    r_array = np.linspace(start=R, stop=lin_max, num=int(self._interp_grid_num / 2))
        #    dr = r_array[2] - r_array[1]
        #    IR_sigma2_ = self._integrand_A15(r_array[1:] - dr/2, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        #    IR_sigma2_dr_lin = IR_sigma2_ * dr
        #    # logarithmic integral for larger extent
        #    max_log = np.log10(max_integrate)
        #    r_array = np.logspace(np.log10(lin_max), max_log, int(self._interp_grid_num / 2))
        #    dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
        #    IR_sigma2_ = self._integrand_A15(r_array, R_, kwargs_mass, kwargs_light, kwargs_anisotropy)
        #    IR_sigma2_dr_log = IR_sigma2_ * dlog_r * r_array
        #    IR_sigma2_dr = np.append(IR_sigma2_dr_lin, IR_sigma2_dr_log)
        if self._log_int is True:
            min_log = np.log10(R)
            max_log = np.log10(max_integrate)
            dlogr = (max_log - min_log) / (self._interp_grid_num - 1)
            r_array = np.logspace(min_log + dlogr / 2., max_log + dlogr / 2., self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            IR_sigma2_ = self._integrand_A15(r_array, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            IR_sigma2_dr = IR_sigma2_ * dlog_r * r_array
        else:
            r_array = np.linspace(start=R, stop=self._max_interpolate, num=self._interp_grid_num)
            dr = r_array[2] - r_array[1]
            IR_sigma2_ = self._integrand_A15(r_array + dr / 2., R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            IR_sigma2_dr = IR_sigma2_ * dr

        IR_sigma2 = np.sum(IR_sigma2_dr) # integral from angle to physical scales
        IR = self.lightProfile.light_2d_finite(R, kwargs_light)
        return IR_sigma2 * 2 * const.G / (const.arcsec * self.cosmo.dd * const.Mpc), IR

    def _I_R_sigma2_interp(self, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        equation A15 in Mamon&Lokas 2005 as interpolation in log space

        :param R: projected radius
        :param kwargs_mass: mass profile keyword arguments
        :param kwargs_light: light model keyword arguments
        :param kwargs_anisotropy: stellar anisotropy keyword arguments
        :return:
        """
        R = np.maximum(R, self._min_integrate)
        if not hasattr(self, '_interp_I_R_sigma2'):
            min_log = np.log10(self._min_integrate)
            max_log = np.log10(self._max_integrate)
            R_array = np.logspace(min_log, max_log, self._interp_grid_num)  # self._interp_grid_num
            I_R_sigma2_array = []
            I_R_array = []
            for R_i in R_array:
                I_R_sigma2_, IR_ = self._I_R_sigma2(R_i, kwargs_mass, kwargs_light, kwargs_anisotropy)
                I_R_sigma2_array.append(I_R_sigma2_)
                I_R_array.append(IR_)
            self._interp_I_R_sigma2 = interp1d(np.log(R_array), np.array(I_R_sigma2_array), fill_value="extrapolate")
            self._interp_I_R = interp1d(np.log(R_array), np.array(I_R_array), fill_value="extrapolate")
        return self._interp_I_R_sigma2(np.log(R)), self._interp_I_R(np.log(R))

    def _integrand_A15(self, r, R, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A15 (in log space) in Mamon&Lokas 2005

        :param r: 3d radius in arc seconds
        :param R: 2d projected radius
        :param kwargs_mass: mass model parameters (following lenstronomy lens model conventions)
        :param kwargs_light: deflector light parameters (following lenstronomy light model conventions)
        :param kwargs_anisotropy: anisotropy parameters, may vary according to anisotropy type chosen.
            We refer to the Anisotropy() class for details on the parameters.
        :return: integrand
        """
        k_r = self.K(r, R, **kwargs_anisotropy)
        #l_r = self.lightProfile.light_3d_interp(r, kwargs_light)
        #m_r = self._mass_3d_interp(r, kwargs_mass)
        l_r = self.lightProfile.light_3d(r, kwargs_light)
        m_r = self.mass_3d(r, kwargs_mass)
        out = k_r * l_r * m_r / r
        return out

    def _jeans_solution_integral(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        interpolated solution of the integral \int_r^{\infty} f(s) l(s) G M(s) / s^2 ds

        :param r: 3d radius
        :param kwargs_mass: mass profile keyword arguments
        :param kwargs_light: light profile keyword arguments
        :param kwargs_anisotropy: anisotropy keyword arguments
        :return: interpolated solution of the Jeans integral
        (copped values at large radius as they become numerically inaccurate)
        """
        if not hasattr(self, '_interp_jeans_integral'):
            min_log = np.log10(self._min_integrate)
            max_log = np.log10(self._max_integrate)  # we extend the integral but ignore these outer solutions in the interpolation
            r_array = np.logspace(min_log, max_log, self._interp_grid_num)
            dlog_r = (np.log10(r_array[2]) - np.log10(r_array[1])) * np.log(10)
            integrand_jeans = self._integrand_jeans_solution(r_array, kwargs_mass, kwargs_light, kwargs_anisotropy) * dlog_r * r_array
            #flip array from inf to finite
            integral_jeans_r = np.cumsum(np.flip(integrand_jeans))
            #flip array back
            integral_jeans_r = np.flip(integral_jeans_r)
            #call 1d interpolation function
            self._interp_jeans_integral = interp1d(np.log(r_array[r_array <= self._max_integrate]),
                                                   integral_jeans_r[r_array <= self._max_integrate],
                                                   fill_value="extrapolate")
        return self._interp_jeans_integral(np.log(r))

    def _integrand_jeans_solution(self, r, kwargs_mass, kwargs_light, kwargs_anisotropy):
        """
        integrand of A1 (in log space) in Mamon&Lokas 2005 to calculate the Jeans equation numerically
        f(s) l(s) M(s) / s^2

        :param r: 3d radius
        :param kwargs_mass: mass model keyword arguments
        :param kwargs_light: light model keyword arguments
        :param kwargs_anisotropy: anisotropy model keyword argument
        :return: integrand value
        """
        f_r = self.anisotropy_solution(r, **kwargs_anisotropy)
        l_r = self.lightProfile.light_3d(r, kwargs_light)
        m_r = self._mass_3d_interp(r, kwargs_mass)
        out = f_r * l_r * m_r / r**2
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
            mass_3d_array[mass_3d_array < 10. ** (-100)] = 10. ** (-100)
            self._log_mass_3d = interp1d(np.log(r_array), np.log(mass_3d_array/r_array),
                                         fill_value=(np.log(mass_3d_array[0] / r_array[0]), -1000), bounds_error=False)
        return np.exp(self._log_mass_3d(np.log(r))) * np.minimum(r, self._max_interpolate)
