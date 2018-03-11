import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussian_kappa
import lenstronomy.Util.param_util as param_util


class TestLensAnalysis(object):

    def setup(self):
        pass

    def test_half_light_radius(self):
        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)
        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': e12, 'e2': e22, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'center_x': -0.01,
            'center_y': 0.9, 'Ra': 0.020000382843298824, 'e1': e1, 'e2': e2,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius_lens(kwargs_profile, numPix=1000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_half_light_radius_hernquist(self):
        Rs = 1.
        kwargs_profile = [{'Rs': Rs, 'sigma0': 1.}]
        kwargs_options = {'lens_model_list': ['NONE'], 'lens_light_model_list': ['HERNQUIST']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = Rs / 0.551
        r_eff = lensAnalysis.half_light_radius_lens(kwargs_profile, numPix=2000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_half_light_radius_source(self):
        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)

        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': e12, 'e2': e22, 'center_x': 0,
            'center_y': 0, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'e1': e1, 'e2': e2, 'center_x': 0,
            'center_y': 0, 'Ra': 0.020000382843298824,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['NONE'], 'source_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius_source(kwargs_profile, numPix=1000, deltaPix=0.05)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_multi_gaussian_lens_light(self):
        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)

        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': e12, 'e2': e22, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304,  'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743,'e1': e1, 'e2': e2, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigma = lensAnalysis.multi_gaussian_lens_light(kwargs_profile, n_comp=20)
        mge = MultiGaussian()
        flux = mge.function(1., 1, amp=amplitudes, sigma=sigma)
        npt.assert_almost_equal(flux, 0.04531989512955493, decimal=8)

    def test_multi_gaussian_lens(self):
        kwargs_options = {'lens_model_list': ['SPEP']}
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.9)
        kwargs_lens = [{'gamma': 1.8, 'theta_E': 0.6, 'e1': e1, 'e2': e2, 'center_x': 0.5, 'center_y': -0.1}]
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigmas, center_x, center_y = lensAnalysis.multi_gaussian_lens(kwargs_lens, n_comp=20)
        model = MultiGaussian_kappa()
        x = np.logspace(-2, 0.5, 10) + 0.5
        y = np.zeros_like(x) - 0.1
        f_xx, f_yy, fxy = model.hessian(x, y, amplitudes, sigmas, center_x=0.5, center_y=-0.1)
        kappa_mge = (f_xx + f_yy) / 2
        kappa_true = lensAnalysis.LensModel.kappa(x, y, kwargs_lens)
        print(kappa_true/kappa_mge)
        for i in range(len(x)):
            npt.assert_almost_equal(kappa_mge[i]/kappa_true[i], 1, decimal=1)

    def test_flux_components(self):
        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)

        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': e12, 'e2': e22, 'center_x': -0.019983826426838536,
                           'center_y': 0.90000011282957304, 'sigma0': 1.3168943578511678},
                          {'Rs': 0.29187068596715743, 'e1': e1, 'e2': e2, 'center_x': 0.020568531548241405,
                           'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824,
                           'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True],
                          'lens_light_model_internal_bool': [True, True],
                          'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        flux_list, R_h_list = lensAnalysis.flux_components(kwargs_profile, n_grid=400, delta_grid=0.01, deltaPix=1., type="lens")
        assert len(flux_list) == 2
        npt.assert_almost_equal(flux_list[0], 0.23898248741810812, decimal=8)
        npt.assert_almost_equal(flux_list[1], 3.0565768930826662, decimal=8)

        kwargs_profile = [{'mean': 1.}]
        kwargs_options = {'lens_light_model_list': ['UNIFORM'], 'lens_model_list': ['NONE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        flux_list, R_h_list = lensAnalysis.flux_components(kwargs_profile, n_grid=400, delta_grid=0.01, deltaPix=1., type="lens")
        assert len(flux_list) == 1
        npt.assert_almost_equal(flux_list[0], 16, decimal=8)


if __name__ == '__main__':
    pytest.main()
