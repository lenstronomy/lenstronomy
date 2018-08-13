import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Analysis.lens_analysis import LensAnalysis
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian, MultiGaussianEllipse
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.util as util


class TestLensAnalysis(object):

    def setup(self):
        pass

    def test_half_light_radius(self):
        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)
        center_x = -0.019983826426838536
        center_y = 0.90000011282957304
        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': e12, 'e2': e22, 'center_x': center_x,
            'center_y': center_y, 'amp': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'center_x': center_x,
            'center_y': center_y, 'Ra': 0.020000382843298824, 'e1': e1, 'e2': e2,
            'amp': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = 0.2878071406838706
        r_eff = lensAnalysis.half_light_radius_lens(kwargs_profile, center_x=center_x, center_y=center_y, numPix=1000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_half_light_radius_hernquist(self):
        Rs = 1.
        kwargs_profile = [{'Rs': Rs, 'amp': 1.}]
        kwargs_options = {'lens_model_list': [], 'lens_light_model_list': ['HERNQUIST']}
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
            'center_y': 0, 'amp': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'e1': e1, 'e2': e2, 'center_x': 0,
            'center_y': 0, 'Ra': 0.020000382843298824,
            'amp': 85.948773973262391}]
        kwargs_options = {'lens_model_list': [], 'source_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius_source(kwargs_profile, numPix=1000, deltaPix=0.05)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

    def test_multi_gaussian_lens_light(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'e1': 0, 'e2': 0, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304,  'amp': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'e1': 0, 'e2': 0, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'Ra': 0.020000382843298824,
            'amp': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigma, center_x, center_y = lensAnalysis.multi_gaussian_lens_light(kwargs_profile, n_comp=20)
        mge = MultiGaussian()
        flux = mge.function(1., 1, amp=amplitudes, sigma=sigma, center_x=center_x, center_y=center_y)
        flux_true = lensAnalysis.LensLightModel.surface_brightness(1, 1, kwargs_profile)
        npt.assert_almost_equal(flux/flux_true, 1, decimal=2)

    def test_mge_lens_light_elliptical(self):
        e1, e2 = 0.3, 0.
        kwargs_profile = [{'amp': 1., 'sigma': 2, 'center_x': 0., 'center_y': 0, 'e1': e1, 'e2': e2}]
        kwargs_options = {'lens_light_model_list': ['GAUSSIAN_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigma, center_x, center_y = lensAnalysis.multi_gaussian_lens_light(kwargs_profile, n_comp=20, e1=e1,
                                                                                       e2=e2, deltaPix=0.05, numPix=400)
        mge = MultiGaussianEllipse()
        flux = mge.function(1., 1, amp=amplitudes, sigma=sigma, center_x=center_x, center_y=center_y, e1=e1, e2=e2)
        flux_true = lensAnalysis.LensLightModel.surface_brightness(1., 1., kwargs_profile)
        npt.assert_almost_equal(flux / flux_true, 1, decimal=1)

    def test_multi_gaussian_lens(self):
        kwargs_options = {'lens_model_list': ['SPEP']}
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.9)
        kwargs_lens = [{'gamma': 1.8, 'theta_E': 0.6, 'e1': e1, 'e2': e2, 'center_x': 0.5, 'center_y': -0.1}]
        lensAnalysis = LensAnalysis(kwargs_options)
        amplitudes, sigmas, center_x, center_y = lensAnalysis.multi_gaussian_lens(kwargs_lens, n_comp=20)
        model = MultiGaussianKappa()
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
                           'center_y': 0.90000011282957304, 'amp': 1.3168943578511678},
                          {'Rs': 0.29187068596715743, 'e1': e1, 'e2': e2, 'center_x': 0.020568531548241405,
                           'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824,
                           'amp': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEP'], 'lens_model_internal_bool': [True],
                          'lens_light_model_internal_bool': [True, True],
                          'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options)
        flux_list, R_h_list = lensAnalysis.flux_components(kwargs_profile, n_grid=400, delta_grid=0.01, deltaPix=1., type="lens")
        assert len(flux_list) == 2
        npt.assert_almost_equal(flux_list[0], 0.23898248741810812, decimal=8)
        npt.assert_almost_equal(flux_list[1], 3.0565768930826662, decimal=8)

        kwargs_profile = [{'amp': 1.}]
        kwargs_options = {'lens_light_model_list': ['UNIFORM'], 'lens_model_list': []}
        lensAnalysis = LensAnalysis(kwargs_options)
        flux_list, R_h_list = lensAnalysis.flux_components(kwargs_profile, n_grid=400, delta_grid=0.01, deltaPix=1., type="lens")
        assert len(flux_list) == 1
        npt.assert_almost_equal(flux_list[0], 16, decimal=8)

    def test_light2mass_conversion(self):
        numPix = 100
        deltaPix = 0.05
        kwargs_options = {'lens_light_model_internal_bool': [True, True],
                          'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC']}
        kwargs_lens_light = [{'R_sersic': 0.5, 'n_sersic': 4, 'amp': 2, 'e1': 0, 'e2': 0.05},
                             {'R_sersic': 1.5, 'n_sersic': 1, 'amp': 2}]
        lensAnalysis = LensAnalysis(kwargs_options)
        kwargs_interpol = lensAnalysis.light2mass_interpol(lens_light_model_list=['SERSIC_ELLIPSE', 'SERSIC'],
                                                                                          kwargs_lens_light=kwargs_lens_light, numPix=numPix, deltaPix=deltaPix, subgrid_res=5)
        from lenstronomy.LensModel.lens_model import LensModel
        lensModel = LensModel(lens_model_list=['INTERPOL_SCALED'])
        kwargs_lens = [kwargs_interpol]
        import lenstronomy.Util.util as util
        x_grid, y_grid = util.make_grid(numPix, deltapix=deltaPix)
        kappa = lensModel.kappa(x_grid, y_grid, kwargs=kwargs_lens)
        kappa = util.array2image(kappa)
        kappa /= np.mean(kappa)
        flux = lensAnalysis.LensLightModel.surface_brightness(x_grid, y_grid, kwargs_lens_light)
        flux = util.array2image(flux)
        flux /= np.mean(flux)
        #import matplotlib.pyplot as plt
        #plt.matshow(flux-kappa)
        #plt.colorbar()
        #plt.show()
        delta_kappa = (kappa - flux)/flux
        max_delta = np.max(np.abs(delta_kappa))
        assert max_delta < 0.1
        #assert max_diff < 0.01
        npt.assert_almost_equal(flux[0, 0], kappa[0, 0], decimal=2)

    def test_light2mass_mge(self):
        from lenstronomy.LightModel.Profiles.gaussian import MultiGaussianEllipse
        multiGaussianEllipse = MultiGaussianEllipse()
        x_grid, y_grid = util.make_grid(numPix=100, deltapix=0.05)
        kwargs_light = [{'amp': [2, 1], 'sigma': [0.1, 1], 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0}]
        light_model_list = ['MULTI_GAUSSIAN_ELLIPSE']
        lensAnalysis = LensAnalysis(kwargs_model={'lens_light_model_list': light_model_list})
        kwargs_mge = lensAnalysis.light2mass_mge(kwargs_lens_light=kwargs_light, numPix=100, deltaPix=0.05, elliptical=True)
        npt.assert_almost_equal(kwargs_mge['e1'], kwargs_light[0]['e1'], decimal=2)

    def test_light2mass_mge_elliptical_sersic(self):
        # same test as above but with Sersic ellipticity definition
        lens_light_kwargs = [
            {'R_sersic': 1.3479852771734446, 'center_x': -0.0014089381116285044, 'n_sersic': 2.260502794737016,
             'amp': 0.08679965264978318, 'center_y': 0.0573684892835563, 'e1': 0.22781838418202335,
             'e2': 0.03841125245832406},
            {'R_sersic': 0.20907637464009315, 'center_x': -0.0014089381116285044, 'n_sersic': 3.0930684763455156,
             'amp': 3.2534559112899633, 'center_y': 0.0573684892835563, 'e1': 0.0323604434989261,
             'e2': -0.12430547471424626}]
        light_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
        lensAnalysis = LensAnalysis({'lens_light_model_list': light_model_list})
        kwargs_mge = lensAnalysis.light2mass_mge(lens_light_kwargs, model_bool_list=None, elliptical=True, numPix=500,
                                                 deltaPix=0.5)
        print(kwargs_mge)
        npt.assert_almost_equal(kwargs_mge['e1'], 0.22, decimal=2)

    def test_ellipticity_lens_light(self):
        e1_in = 0.1
        e2_in = 0
        kwargs_light = [{'amp': 1, 'sigma': 1., 'center_x': 0, 'center_y': 0, 'e1': e1_in, 'e2': e2_in}]
        light_model_list = ['GAUSSIAN_ELLIPSE']
        lensAnalysis = LensAnalysis(kwargs_model={'lens_light_model_list': light_model_list})
        e1, e2 = lensAnalysis.ellipticity_lens_light(kwargs_light, center_x=0, center_y=0, model_bool_list=None, deltaPix=0.1,
                               numPix=200)
        npt.assert_almost_equal(e1, e1_in, decimal=4)
        npt.assert_almost_equal(e2, e2_in, decimal=4)

        #SERSIC
        e1_in = 0.1
        e2_in = 0
        kwargs_light = [{'amp': 1, 'n_sersic': 2., 'R_sersic': 1, 'center_x': 0,'center_y': 0, 'e1': e1_in, 'e2': e2_in}]
        light_model_list = ['SERSIC_ELLIPSE']
        lensAnalysis = LensAnalysis(kwargs_model={'lens_light_model_list': light_model_list})
        e1, e2 = lensAnalysis.ellipticity_lens_light(kwargs_light, center_x=0, center_y=0, model_bool_list=None, deltaPix=0.2,
                               numPix=400)
        print(e1, e2)
        npt.assert_almost_equal(e1, e1_in, decimal=3)
        npt.assert_almost_equal(e2, e2_in, decimal=3)

    def test_mass_fraction_within_radius(self):
        center_x, center_y = 0.5, -1
        theta_E = 1.1
        kwargs_lens = [{'theta_E': 1.1, 'center_x': center_x, 'center_y': center_y}]
        lensAnalysis = LensAnalysis(kwargs_model={'lens_model_list': ['SIS']})
        kappa_mean_list = lensAnalysis.mass_fraction_within_radius(kwargs_lens, center_x, center_y, theta_E, numPix=100)
        npt.assert_almost_equal(kappa_mean_list[0], 1, 2)


if __name__ == '__main__':
    pytest.main()
