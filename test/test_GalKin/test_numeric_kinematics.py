"""
Tests for `Galkin` module.
"""
import pytest
import numpy as np
import numpy.testing as npt

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
        r_eff = 1.5
        Rs = 0.551 * r_eff
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_mass = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 500, 'log_integration': True,
                               'max_integrate': 100}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        analytic_kin = AnalyticKinematics(kwargs_cosmo)
        numeric_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
        rho0_r0_gamma = analytic_kin._rho0_r0_gamma(theta_E, gamma)
        r_array = np.logspace(-2, 0.5, 10)
        sigma_r_analytic_array = []
        sigma_r_num_array = []
        for r in r_array:
            sigma_r2_analytic = analytic_kin._sigma_r2(r=r, a=Rs, gamma=gamma, r_ani=r_ani, rho0_r0_gamma=rho0_r0_gamma)
            sigma_r2_num = numeric_kin.sigma_r2(r, kwargs_mass, kwargs_light, kwargs_anisotropy)
            sigma_r_analytic = np.sqrt(sigma_r2_analytic) / 1000
            sigma_r_num = np.sqrt(sigma_r2_num) / 1000
            sigma_r_num_array.append(sigma_r_num)
            sigma_r_analytic_array.append(sigma_r_analytic)
        #import matplotlib.pyplot as plt
        #plt.semilogx(r_array, np.array(sigma_r_analytic_array)/np.array(sigma_r_num_array), label='analytic')
        #plt.semilogx(r_array, sigma_r_num_array, label='numeric')
        #plt.legend()
        #plt.show()
        npt.assert_almost_equal(sigma_r_num_array, sigma_r_analytic_array, decimal=-1)

    def test_delete_cache(self):
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'interpol_grid_num': 500, 'log_integration': True,
                           'max_integrate': 100}

        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 1}
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
