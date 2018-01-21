import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian


class TestLensAnalysis(object):

    def setup(self):
        pass

    def test_half_light_radius(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options, {})
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius_lens(kwargs_profile, numPix=1000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_half_light_radius_hernquist(self):
        Rs = 1.
        kwargs_profile = [{'Rs': Rs, 'sigma0': 1.}]
        kwargs_options = {'lens_model_list': ['NONE'], 'lens_light_model_list': ['HERNQUIST']}
        lensAnalysis = LensAnalysis(kwargs_options, {})
        r_eff_true = Rs / 0.551
        r_eff = lensAnalysis.half_light_radius_lens(kwargs_profile, numPix=2000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_multi_gaussian_lens_light(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options, {})
        amplitudes, sigma = lensAnalysis.multi_gaussian_lens_light(kwargs_profile, n_comp=20)
        mge = MultiGaussian()
        flux = mge.function(1., 1, amp=amplitudes, sigma=sigma)
        npt.assert_almost_equal(flux,0.04604026574172369, decimal=8)


if __name__ == '__main__':
    pytest.main()