"""
Tests for `Galkin` module.
"""
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Util import constants as const
import scipy.integrate as integrate

from lenstronomy.GalKin.numeric_kinematics import NumericKinematics
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics


class TestMassProfile(object):

    def setup(self):
        pass

    def test_mass_3d(self):
        kwargs_model = {'mass_profile_list': ['HERNQUIST'], 'light_profile_list': ['HERNQUIST'],
                        'anisotropy_model': 'isotropic'}
        massProfile = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo={'d_d': 1., 'd_s': 2., 'd_ds': 1.})
        r = 0.3
        kwargs_profile = [{'sigma0': 1., 'Rs': 0.5}]
        mass_3d = massProfile._mass_3d_interp(r, kwargs_profile)
        mass_3d_exact = massProfile.mass_3d(r, kwargs_profile)
        npt.assert_almost_equal(mass_3d/mass_3d_exact, 1., decimal=3)

    def test_sigma_r2(self):
        """
        tests the solution of the Jeans equation for sigma**2(r), where r is the 3d radius.
        Test is compared to analytic OM solution with power-law and Hernquist light profile

        :return:
        """
        light_profile_list = ['HERNQUIST']
        r_eff = 0.5
        Rs = 0.551 * r_eff
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 1.95
        kwargs_mass = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 0.5
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 2000, 'log_integration': True,
                               'max_integrate': 4000, 'min_integrate': 0.001}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        analytic_kin = AnalyticKinematics(kwargs_cosmo, **kwargs_numerics)
        numeric_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
        rho0_r0_gamma = analytic_kin._rho0_r0_gamma(theta_E, gamma)
        r_array = np.logspace(-2.9, 2.9, 100)
        sigma_r_analytic_array = []
        sigma_r_num_array = []
        for r in r_array:
            sigma_r2_analytic = analytic_kin._sigma_r2(r=r, a=Rs, gamma=gamma, r_ani=r_ani, rho0_r0_gamma=rho0_r0_gamma)
            sigma_r2_num = numeric_kin.sigma_r2(r, kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma_r_analytic = np.sqrt(sigma_r2_analytic) / 1000
            sigma_r_num = np.sqrt(sigma_r2_num) / 1000
            sigma_r_num_array.append(sigma_r_num)
            sigma_r_analytic_array.append(sigma_r_analytic)

        npt.assert_almost_equal(sigma_r_num_array, sigma_r_analytic_array, decimal=-2)
        npt.assert_almost_equal(np.array(sigma_r_num_array) / np.array(sigma_r_analytic_array), 1, decimal=-2)
        print(np.array(sigma_r_num_array) / np.array(sigma_r_analytic_array))

    def test_sigma_s2(self):
        """
        test LOS projected velocity dispersion at 3d ratios (numerical Jeans equation solution vs analytic one)
        """
        light_profile_list = ['HERNQUIST']
        r_eff = 0.5
        Rs = 0.551 * r_eff
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 1.95
        kwargs_mass = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 0.5
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 2000, 'log_integration': True,
                           'max_integrate': 4000, 'min_integrate': 0.001}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        analytic_kin = AnalyticKinematics(kwargs_cosmo, **kwargs_numerics)
        numeric_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
        r_list = np.logspace(-2, 1, 10)
        for r in r_list:
            for R in np.linspace(start=0, stop=r, num=5):
                sigma_s2_analytic, I_R = analytic_kin.sigma_s2(r, R, {'theta_E': theta_E, 'gamma': gamma}, {'r_eff': r_eff}, kwargs_anisotropy)
                sigma_s2_full_num = numeric_kin.sigma_s2_r(r, R, kwargs_mass, kwargs_light, kwargs_anisotropy)
                npt.assert_almost_equal(sigma_s2_full_num/sigma_s2_analytic, 1, decimal=2)

    def test_I_R_sigma_s2(self):
        light_profile_list = ['HERNQUIST']
        r_eff = 1
        Rs = 0.551 * r_eff
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 1.95
        kwargs_mass = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 0.5
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 4000, 'log_integration': True,
                           'max_integrate': 100, 'min_integrate': 0.0001, 'max_light_draw': 50}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        numeric_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)

        # check whether projected light integral is the same as analytic expression
        R = 1
        I_R_sigma2, I_R = numeric_kin._I_R_sigma2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
        out = integrate.quad(lambda x: numeric_kin.lightProfile.light_3d(np.sqrt(R ** 2 + x ** 2), kwargs_light), kwargs_numerics['min_integrate'],
                        np.sqrt(kwargs_numerics['max_integrate']**2 - R**2))
        l_R_quad = out[0] * 2
        npt.assert_almost_equal(l_R_quad / I_R, 1, decimal=2)

        l_R = numeric_kin.lightProfile.light_2d(R, kwargs_light)
        npt.assert_almost_equal(l_R / I_R, 1, decimal=2)

    def test_log_linear_integral(self):
        # light profile
        light_profile_list = ['HERNQUIST']
        Rs = .5
        kwargs_light = [{'Rs':  Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics_linear = {'interpol_grid_num': 2000, 'log_integration': False,
                           'max_integrate': 10, 'min_integrate': 0.001}
        kwargs_numerics_log = {'interpol_grid_num': 1000, 'log_integration': True,
                           'max_integrate': 10, 'min_integrate': 0.001}
        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}

        numerics_linear = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_linear)
        numerics_log = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_log)
        R = np.logspace(-2, 0, 100)

        lin_I_R = np.zeros_like(R)
        log_I_R = np.zeros_like(R)
        for i in range(len(R)):
            lin_I_R[i], _ = numerics_linear._I_R_sigma2(R[i], kwargs_profile, kwargs_light, kwargs_anisotropy)
            log_I_R[i], _ = numerics_log._I_R_sigma2(R[i], kwargs_profile, kwargs_light, kwargs_anisotropy)

        #import matplotlib.pyplot as plt
        #plt.semilogx(R, lin_I_R / log_I_R, 'r', label='lin /log integrate')
        #plt.legend()
        #plt.show()

        R_ = 1
        r_array = np.logspace(start=np.log10(R_+0.001), stop=1, num=100)
        integrad_a15 = numerics_linear._integrand_A15(r_array, R_, kwargs_profile, kwargs_light, kwargs_anisotropy)
        #plt.loglog(r_array, integrad_a15)
        #plt.show()

        for i in range(len(R)):
            npt.assert_almost_equal(log_I_R[i] / lin_I_R[i], 1, decimal=2)

        #assert 1 == 0

    def test_I_R_sigma(self):
        """
        test numerical integral against quad integrator
        :return:
        """
        light_profile_list = ['HERNQUIST']
        Rs = .5
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 2000, 'log_integration': True,
                                  'max_integrate': 1000, 'min_integrate': 0.0001}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}

        numerics = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)

        R = 0.1
        out = integrate.quad(lambda x: numerics._integrand_A15(x, R, kwargs_profile, kwargs_light, kwargs_anisotropy),
                             R, kwargs_numerics['max_integrate'])

        I_R_sigma_quad = out[0] * 2 * const.G / (const.arcsec * kwargs_cosmo['d_d'] * const.Mpc)
        I_R_sigma_numerics_log, _ = numerics._I_R_sigma2(R, kwargs_profile, kwargs_light, kwargs_anisotropy)

        kwargs_numerics_lin = {'interpol_grid_num': 2000, 'log_integration': False,
                           'max_integrate': 1000, 'min_integrate': 0.0001}
        numerics_lin = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_lin)
        I_R_simga_numerics_lin, _ = numerics_lin._I_R_sigma2(R, kwargs_profile, kwargs_light, kwargs_anisotropy)
        npt.assert_almost_equal(I_R_sigma_numerics_log / I_R_sigma_quad, 1, decimal=2)

        # We do not test the linear integral as it is not as accurate!!!
        #npt.assert_almost_equal(I_R_simga_numerics_lin / I_R_simga_quad, 1, decimal=2)


        # here we test the interpolation
        numerics = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        R = 1
        I_R_sigma2_interp, I_R_interp = numerics._I_R_sigma2_interp(R, kwargs_profile, kwargs_light, kwargs_anisotropy)
        I_R_sigma2, I_R = numerics._I_R_sigma2(R, kwargs_profile, kwargs_light, kwargs_anisotropy)
        npt.assert_almost_equal(I_R_sigma2_interp / I_R_sigma2, 1, decimal=3)
        npt.assert_almost_equal(I_R_interp / I_R, 1, decimal=3)

    def test_power_law_test(self):
        # tests a isotropic velocity anisotropy on a singular isothermal sphere with the same tracer particle distribution
        # This should result in a constant velocity dispersion as a function of radius, analytically known

        # set up power-law light profile
        light_model = ['POWER_LAW']
        kwargs_light = [{'gamma': 2, 'amp': 1, 'e1': 0, 'e2': 0}]

        lens_model = ['SIS']
        kwargs_mass = [{'theta_E': 1}]

        anisotropy_type = 'isotropic'
        kwargs_anisotropy = {}
        kwargs_model = {'mass_profile_list': lens_model,
                        'light_profile_list': light_model,
                        'anisotropy_model': anisotropy_type}
        kwargs_numerics = {'interpol_grid_num': 2000, 'log_integration': True,
                                  'max_integrate': 1000, 'min_integrate': 0.0001}
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}

        # compute analytic velocity dispersion of SIS profile

        v_sigma_c2 = kwargs_mass[0]['theta_E'] * const.arcsec / (4 * np.pi) * kwargs_cosmo['d_s'] / kwargs_cosmo['d_ds']
        v_sigma_true = np.sqrt(v_sigma_c2) * const.c / 1000

        numerics = NumericKinematics(kwargs_model=kwargs_model, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        R = 2
        I_R_sigma2, I_R = numerics._I_R_sigma2(R, kwargs_mass, kwargs_light, kwargs_anisotropy={})
        sigma_v = np.sqrt(I_R_sigma2 / I_R) / 1000
        print(sigma_v, v_sigma_true)
        npt.assert_almost_equal(sigma_v / v_sigma_true, 1, decimal=2)

        # plot as radial distance of projected dispersion
        r_array = np.logspace(start=-2, stop=1, num=100)

        sigma_array = np.zeros_like(r_array)
        for i, R in enumerate(r_array):
            I_R_sigma2, I_R = numerics._I_R_sigma2(R, kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma_array[i] = np.sqrt(I_R_sigma2 / I_R) / 1000

        #import matplotlib.pyplot as plt
        #plt.semilogx(r_array, sigma_array)
        #plt.hlines(v_sigma_true, xmin=r_array[0], xmax=r_array[-1])
        #plt.show()

        npt.assert_almost_equal(sigma_array / v_sigma_true, 1, decimal=2)

        # and here we test the 3d radial velocity dispersion solution
        sigma_array = np.zeros_like(r_array)
        R_test = 0
        for i, r in enumerate(r_array):
            sigma_s2 = numerics.sigma_s2_r(r, R_test, kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma_array[i] = np.sqrt(sigma_s2) / 1000
        npt.assert_almost_equal(sigma_array / v_sigma_true, 1, decimal=2)

    def test_delete_cache(self):
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 2000, 'log_integration': True,
                           'max_integrate': 1000, 'min_integrate': 0.0001}
        kwargs_model = {'mass_profile_list': [],
                        'light_profile_list': [],
                        'anisotropy_model': 'const'}
        numeric_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
        numeric_kin._interp_jeans_integral = 1
        numeric_kin._log_mass_3d = 2
        numeric_kin.delete_cache()
        assert hasattr(numeric_kin, '_log_mass_3d') is False
        assert hasattr(numeric_kin, '_interp_jeans_integral') is False


if __name__ == '__main__':
    pytest.main()
