import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.util as util


class TestLensProfileAnalysis(object):

    def setup(self):
        pass

    def test_profile_slope(self):
        lens_model = LensProfileAnalysis(LensModel(lens_model_list=['SPP']))
        gamma_in = 2.
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)
        gamma_in = 1.7
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        gamma_in = 2.5
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        kwargs_lens_bad = [{'theta_E': 100, 'gamma': 2, 'center_x': 0, 'center_y': 0}]
        gamma_out_bad = lens_model.profile_slope(kwargs_lens_bad, verbose=False)
        assert np.isnan(gamma_out_bad)

        lens_model = LensProfileAnalysis(LensModel(lens_model_list=['SPEP']))
        gamma_in = 2.
        phi, q = 0.34403343049704888, 0.89760957136967312
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1.4516812130749424, 'e1': e1, 'e2': e2, 'center_x': -0.04507598845306314,
         'center_y': 0.054491803177414651, 'gamma': gamma_in}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

    def test_effective_einstein_radius(self):
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        lensModel = LensProfileAnalysis(LensModel(lens_model_list=['SIS']))
        ret = lensModel.effective_einstein_radius(kwargs_lens,
                                                  get_precision=True)

        assert len(ret) == 2
        npt.assert_almost_equal(ret[0], 1., decimal=2)
        kwargs_lens_bad = [{'theta_E': 100, 'center_x': 0, 'center_y': 0}]
        ret_nan = lensModel.effective_einstein_radius(kwargs_lens_bad,
                                                      get_precision=True, verbose=False)
        assert np.isnan(ret_nan)

    def test_external_lensing_effect(self):
        lens_model_list = ['SHEAR']
        kwargs_lens = [{'e1': 0.1, 'e2': 0.01}]
        lensModel = LensProfileAnalysis(LensModel(lens_model_list))
        alpha0_x, alpha0_y, kappa_ext, shear1, shear2 = lensModel.external_lensing_effect(kwargs_lens, lens_model_internal_bool=[False])
        print(alpha0_x, alpha0_y, kappa_ext, shear1, shear2)
        assert alpha0_x == 0
        assert alpha0_y == 0
        assert shear1 == 0.1
        assert shear2 == 0.01
        assert kappa_ext == 0

    def test_multi_gaussian_lens(self):
        kwargs_options = {'lens_model_list': ['SPEP']}
        lensModel = LensModel(**kwargs_options)
        lensAnalysis = LensProfileAnalysis(lens_model=lensModel)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.9)
        kwargs_lens = [{'gamma': 1.8, 'theta_E': 0.6, 'e1': e1, 'e2': e2, 'center_x': 0.5, 'center_y': -0.1}]
        amplitudes, sigmas, center_x, center_y = lensAnalysis.multi_gaussian_lens(kwargs_lens, n_comp=20)
        model = MultiGaussianKappa()
        x = np.logspace(-2, 0.5, 10) + 0.5
        y = np.zeros_like(x) - 0.1
        f_xx, f_yy, fxy = model.hessian(x, y, amplitudes, sigmas, center_x=0.5, center_y=-0.1)
        kappa_mge = (f_xx + f_yy) / 2
        kappa_true = lensAnalysis._lens_model.kappa(x, y, kwargs_lens)
        print(kappa_true/kappa_mge)
        for i in range(len(x)):
            npt.assert_almost_equal(kappa_mge[i]/kappa_true[i], 1, decimal=1)

    def test_mass_fraction_within_radius(self):
        center_x, center_y = 0.5, -1
        theta_E = 1.1
        kwargs_lens = [{'theta_E': 1.1, 'center_x': center_x, 'center_y': center_y}]
        lensModel = LensModel(**{'lens_model_list': ['SIS']})
        lensAnalysis = LensProfileAnalysis(lens_model=lensModel)
        kappa_mean_list = lensAnalysis.mass_fraction_within_radius(kwargs_lens, center_x, center_y, theta_E, numPix=100)
        npt.assert_almost_equal(kappa_mean_list[0], 1, 2)

    def test_lens_center(self):
        center_x, center_y = 0.43, -0.67
        kwargs_lens = [{'theta_E': 1, 'center_x': center_x, 'center_y': center_y}]
        lensModel = LensModel(**{'lens_model_list': ['SIS']})
        profileAnalysis = LensProfileAnalysis(lens_model=lensModel)
        center_x_out, center_y_out = profileAnalysis.lens_center(kwargs_lens)
        npt.assert_almost_equal(center_x_out, center_x, 2)
        npt.assert_almost_equal(center_y_out, center_y, 2)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            raise ValueError()


if __name__ == '__main__':
    pytest.main()
