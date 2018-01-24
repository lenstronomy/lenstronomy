__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.multi_plane import MultiLens
from lenstronomy.LensModel.lens_model import LensModel


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        pass

    def test_sis_alpha(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiLens(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        alpha_x_simple, alpha_y_simple = lensModel.alpha(1, 0, kwargs_lens)
        alpha_x_multi, alpha_y_multi = lensModelMutli.alpha(1, 0, kwargs_lens)
        assert alpha_x_simple == alpha_x_multi
        assert alpha_y_simple == alpha_y_multi

    def test_sis_ray_tracing(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiLens(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        beta_x_simple, beta_y_simple = lensModel.ray_shooting(1, 0, kwargs_lens)
        beta_x_multi, beta_y_multi = lensModelMutli.ray_shooting(1, 0, kwargs_lens)
        assert beta_x_simple == beta_x_multi
        assert beta_y_simple == beta_y_multi

    def test_sis_hessian(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiLens(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        f_xx_simple, f_xy_simple, f_yy_simple = lensModel.hessian(1, 0, kwargs_lens)
        f_xx_multi, f_xy_multi, f_yy_multi = lensModelMutli.hessian(1, 0, kwargs_lens, diff=0.000001)
        npt.assert_almost_equal(f_xx_simple, f_xx_multi, decimal=5)
        npt.assert_almost_equal(f_xy_simple, f_xy_multi, decimal=5)
        npt.assert_almost_equal(f_yy_simple, f_yy_multi, decimal=5)

    def test_sis_kappa_gamma_mag(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiLens(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kappa_simple = lensModel.kappa(1, 0, kwargs_lens)
        kappa_multi = lensModelMutli.kappa(1, 0, kwargs_lens)
        npt.assert_almost_equal(kappa_simple, kappa_multi, decimal=5)

        gamma1_simple, gamma2_simple = lensModel.gamma(1, 0, kwargs_lens)
        gamma1_multi, gamma2_multi = lensModelMutli.gamma(1, 0, kwargs_lens)
        npt.assert_almost_equal(gamma1_simple, gamma1_multi, decimal=5)
        npt.assert_almost_equal(gamma2_simple, gamma2_multi, decimal=5)

        mag_simple = lensModel.magnification(0.99, 0, kwargs_lens)
        mag_multi = lensModelMutli.magnification(0.99, 0, kwargs_lens)
        npt.assert_almost_equal(mag_simple, mag_multi, decimal=5)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")