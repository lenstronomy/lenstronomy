import pytest
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics


class TestAnalyticKinematics(object):

    def setup(self):
        pass

    def test_sigma_s2(self):
        kwargs_aperture = {'center_ra': 0, 'width': 1, 'length': 1, 'angle': 0, 'center_dec': 0,
                           'aperture_type': 'slit'}
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 1}
        kin = AnalyticKinematics(kwargs_aperture, kwargs_psf, kwargs_cosmo)
        kwargs_light = {'r_eff': 1}
        sigma_s2 = kin.sigma_s2(r=1, R=0.1, kwargs_mass={'theta_E': 1, 'gamma': 2}, kwargs_light=kwargs_light,
                                kwargs_anisotropy={'r_ani': 1})
        assert 'a' in kwargs_light

    def test_radius_slope_anisotropy(self):
        kwargs_aperture = {'center_ra': 0, 'width': 1, 'length': 1, 'angle': 0, 'center_dec': 0,
                           'aperture_type': 'slit'}
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 1}
        kin = AnalyticKinematics(kwargs_aperture, kwargs_psf, kwargs_cosmo)
        r = 1
        theta_E, gamma = 1, 2.
        a_ani = 10
        r_eff = 0.1
        out = kin.check_df(r, theta_E, gamma, a_ani, r_eff)
        assert out > 0
        print(out)
        #import matplotlib.pyplot as plt
        #import numpy as np
        #r_list = np.logspace(-1, 1, 20)
        #crit_list = []
        #for r in r_list:
        #    crit_list.append(kin.check_df(r, theta_E, gamma, a_ani, r_eff))
        #plt.plot(r_list, crit_list)
        #plt.show()
        #assert 1 == 0


if __name__ == '__main__':
    pytest.main()
