__author__ = 'sibirrer'

import pytest
import numpy.testing as npt

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.numeric_lens_differentials import NumericLens


class TestNumerics(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.lensModel = LensModel(['GAUSSIAN'])
        self.lensModelNum = NumericLens(['GAUSSIAN'])
        self.kwargs = [{'amp': 1./4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}]

    def test_kappa(self):
        x, y = 1., 1.
        output = self.lensModel.kappa(x, y, self.kwargs)
        output_num = self.lensModelNum.kappa(x, y, self.kwargs)
        npt.assert_almost_equal(output_num, output, decimal=5)

    def test_gamma(self):
        x, y = 1., 1.
        output1, output2 = self.lensModel.gamma(x, y, self.kwargs)
        output1_num, output2_num = self.lensModelNum.gamma(x, y, self.kwargs)
        npt.assert_almost_equal(output1_num, output1, decimal=5)
        npt.assert_almost_equal(output2_num, output2, decimal=5)

    def test_magnification(self):
        x, y = 1., 1.
        output = self.lensModel.magnification(x, y, self.kwargs)
        output_num = self.lensModelNum.magnification(x, y, self.kwargs)
        npt.assert_almost_equal(output_num, output, decimal=5)

    def test_differentials(self):
        x, y = 1., 1.
        f_xx, f_xy, f_yy = self.lensModel.hessian(x, y, self.kwargs)
        f_xx_num, f_xy_num, f_yx_num, f_yy_num = self.lensModelNum.hessian(x, y, self.kwargs)
        assert f_xy_num == f_yx_num
        npt.assert_almost_equal(f_xx_num, f_xx, decimal=5)
        npt.assert_almost_equal(f_xy_num, f_xy, decimal=5)
        npt.assert_almost_equal(f_yy_num, f_yy, decimal=5)


class TestNumericsProfile(object):
    """
    tests the second derivatives of various lens models
    """
    def setup(self):
        pass

    def assert_differentials(self, lens_model, kwargs):
        lensModelNum = NumericLens(lens_model)
        x, y = 1., 2.
        lensModel = LensModel(lens_model)
        f_x, f_y = lensModel.lens_model.alpha(x, y, [kwargs])
        f_xx, f_xy, f_yy = lensModel.hessian(x, y, [kwargs])
        f_x_num, f_y_num = lensModelNum.lens_model.alpha(x, y, [kwargs])
        f_xx_num, f_xy_num, f_yx_num, f_yy_num = lensModelNum.hessian(x, y, [kwargs])
        print(f_xx_num, f_xx)
        print(f_yy_num, f_yy)
        print(f_xy_num, f_xy)
        print((f_xx - f_yy)**2/4 + f_xy**2, (f_xx_num - f_yy_num)**2/4 + f_xy_num**2)
        npt.assert_almost_equal(f_x, f_x_num, decimal=5)
        npt.assert_almost_equal(f_y, f_y_num, decimal=5)
        npt.assert_almost_equal(f_xx, f_xx_num, decimal=3)
        npt.assert_almost_equal(f_yy, f_yy_num, decimal=3)
        npt.assert_almost_equal(f_xy, f_xy_num, decimal=3)

        x, y = 1., 0.
        f_x, f_y = lensModel.lens_model.alpha(x, y, [kwargs])
        f_xx, f_xy, f_yy = lensModel.hessian(x, y, [kwargs])
        f_x_num, f_y_num = lensModelNum.lens_model.alpha(x, y, [kwargs])
        f_xx_num, f_xy_num, f_yx_num, f_yy_num = lensModelNum.hessian(x, y, [kwargs])
        print(f_xx_num, f_xx)
        print(f_yy_num, f_yy)
        print(f_xy_num, f_xy)
        print((f_xx - f_yy)**2/4 + f_xy**2, (f_xx_num - f_yy_num)**2/4 + f_xy_num**2)
        npt.assert_almost_equal(f_x, f_x_num, decimal=5)
        npt.assert_almost_equal(f_y, f_y_num, decimal=5)
        npt.assert_almost_equal(f_xx, f_xx_num, decimal=3)
        npt.assert_almost_equal(f_yy, f_yy_num, decimal=3)
        npt.assert_almost_equal(f_xy, f_xy_num, decimal=3)

    def test_gaussian(self):
        lens_model = ['GAUSSIAN']
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}
        self.assert_differentials(lens_model, kwargs)
        kwargs = {'amp': 1. / 4., 'sigma_x': 20., 'sigma_y': 20., 'center_x': 0., 'center_y': 0.}
        self.assert_differentials(lens_model, kwargs)
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 2., 'center_y': 2.}
        self.assert_differentials(lens_model, kwargs)

    def test_gausian_kappa(self):
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}
        lens_model = ['GAUSSIAN_KAPPA']
        self.assert_differentials(lens_model, kwargs)

    def test_external_shear(self):
        kwargs = {'e1': 0.1, 'e2': -0.1}
        lens_model = ['SHEAR']
        self.assert_differentials(lens_model, kwargs)

    def test_sis(self):
        kwargs = {'theta_E': 0.5}
        lens_model = ['SIS']
        self.assert_differentials(lens_model, kwargs)

    def test_flexion(self):
        kwargs = {'g1': 0.01, 'g2': -0.01, 'g3': 0.001, 'g4': 0}
        lens_model = ['FLEXION']
        self.assert_differentials(lens_model, kwargs)

    def test_nfw(self):
        kwargs = {'theta_Rs': .1, 'Rs': 5.}
        lens_model = ['NFW']
        self.assert_differentials(lens_model, kwargs)

    def test_nfw_ellipse(self):
        kwargs = {'theta_Rs': .1, 'Rs': 5., 'q': 0.9, 'phi_G': 0.}
        lens_model = ['NFW_ELLIPSE']
        self.assert_differentials(lens_model, kwargs)

    def test_point_mass(self):
        kwargs = {'theta_E': 1.}
        lens_model = ['POINT_MASS']
        self.assert_differentials(lens_model, kwargs)

    def test_sersic(self):
        kwargs = {'n_sersic': .5, 'r_eff': 1.5, 'k_eff': 0.3}
        lens_model = ['SERSIC']
        self.assert_differentials(lens_model, kwargs)

    def test_sersic_ellipse(self):
        kwargs = {'n_sersic': 2., 'r_eff': 0.5, 'k_eff': 0.3, 'q': 0.9, 'phi_G': 1.}
        lens_model = ['SERSIC_ELLIPSE']
        self.assert_differentials(lens_model, kwargs)

    def test_shapelets_pot_2(self):
        kwargs = {'coeffs': [0, 1, 2, 3, 4, 5], 'beta': 0.3}
        lens_model = ['SHAPELETS_CART']
        self.assert_differentials(lens_model, kwargs)

    def test_sis_truncate(self):
        kwargs = {'theta_E': 0.5, 'r_trunc': 2.}
        lens_model = ['SIS_TRUNCATED']
        self.assert_differentials(lens_model, kwargs)

    def test_spep(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.}
        lens_model = ['SPEP']
        self.assert_differentials(lens_model, kwargs)

    def test_spp(self):
        kwargs = {'theta_E': 0.5, 'gamma': 1.9}
        lens_model = ['SPP']
        self.assert_differentials(lens_model, kwargs)

    def test_PJaffe(self):
        kwargs = {'sigma0': 1., 'Ra': 0.2, 'Rs': 2.}
        lens_model = ['PJAFFE']
        self.assert_differentials(lens_model, kwargs)

    def test_PJaffe_ellipse(self):
        kwargs = {'sigma0': 1., 'Ra': 0.2, 'Rs': 2., 'q': .8, 'phi_G': 1.}
        lens_model = ['PJAFFE_ELLIPSE']
        self.assert_differentials(lens_model, kwargs)

    def test_Hernquist(self):
        kwargs = {'sigma0': 1., 'Rs': 1.5}
        lens_model = ['HERNQUIST']
        self.assert_differentials(lens_model, kwargs)

    def test_Hernquist_ellipse(self):
        kwargs = {'sigma0': 1., 'Rs': 1.5, 'q': 0.8, 'phi_G': 1.}
        lens_model = ['HERNQUIST_ELLIPSE']
        self.assert_differentials(lens_model, kwargs)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")