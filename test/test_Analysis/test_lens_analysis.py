import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussian_kappa


class TestLensAnalysis(object):

    def setup(self):
        pass

    def test_half_light_radius(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': -0.01,
            'center_y': 0.9, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
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
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': 0,
            'center_y': 0, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0,
            'center_y': 0, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['NONE'], 'source_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius_source(kwargs_profile, numPix=1000, deltaPix=0.05)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_multi_gaussian_lens_light(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigma = lensAnalysis.multi_gaussian_lens_light(kwargs_profile, n_comp=20)
        mge = MultiGaussian()
        flux = mge.function(1., 1, amp=amplitudes, sigma=sigma)
        npt.assert_almost_equal(flux, 0.04531989512955493, decimal=8)

    def test_multi_gaussian_lens(self):
        kwargs_options = {'lens_model_list': ['SPEP']}
        kwargs_lens = [{'gamma': 1.8, 'theta_E': 0.6, 'q': 0.9, 'phi_G': 0, 'center_x': 0.5, 'center_y': -0.1}]
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
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
                           'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
                          {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': -0.01,
                           'center_y': 0.9, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
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

    def test_buldge_disk_ratio(self):
        kwargs_buldge_disk = {'I0_b': 10, 'R_b': 0.1, 'phi_G_b': 0, 'q_b': 1, 'I0_d': 2, 'R_d': 1, 'phi_G_d': 0.5, 'q_d': 0.7, 'center_x': 0, 'center_y': 0}
        light_tot, light_buldge = LensAnalysis.buldge_disk_ratio(kwargs_buldge_disk)
        npt.assert_almost_equal(light_buldge/light_tot, 0.108, decimal=2)


if __name__ == '__main__':
    pytest.main()