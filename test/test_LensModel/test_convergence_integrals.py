from lenstronomy.LensModel import convergence_integrals
import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.sis import SIS
import numpy.testing as npt
import pytest


class TestConvergenceIntegrals(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        pass

    def test_potential_from_kappa(self):

        sis = SIS()
        deltaPix = 0.005
        x_grid, y_grid = util.make_grid(numPix=2000, deltapix=deltaPix)
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        f_ = sis.function(x_grid, y_grid, **kwargs_sis)
        f_ = util.array2image(f_)
        kappa = util.array2image((f_xx + f_yy) / 2.)
        potential_num = convergence_integrals.potential_from_kappa_grid(kappa, deltaPix)

        x1, y1 = 560, 500
        x2, y2 = 550, 500
        # test relative potential at two different point way inside the kappa map
        d_f_num = potential_num[x1, y1] - potential_num[x2, y2]
        d_f = f_[x1, y1] - f_[x2, y2]
        npt.assert_almost_equal(d_f_num, d_f, decimal=2)

    def test_potential_from_kappa_adaptiv(self):
        sis = SIS()
        deltaPix = 0.01
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}
        low_res_factor = 5
        high_res_kernel_size = 5
        x_grid, y_grid = util.make_grid(numPix=1000, deltapix=deltaPix)

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        kappa = util.array2image((f_xx + f_yy) / 2.)
        f_num = convergence_integrals.potential_from_kappa_grid_adaptive(kappa, deltaPix, low_res_factor, high_res_kernel_size)

        x_grid_low, y_grid_low = util.make_grid(numPix=1000/low_res_factor, deltapix=deltaPix*low_res_factor)
        f_low = sis.function(x_grid_low, y_grid_low, **kwargs_sis)
        f_low = util.array2image(f_low)
        x1, y1 = 56, 50
        x2, y2 = 55, 50
        # test relative potential at two different point way inside the kappa map
        d_f_num = f_num[x1, y1] - f_num[x2, y2]
        d_f = f_low[x1, y1] - f_low[x2, y2]
        npt.assert_almost_equal(d_f_num, d_f, decimal=2)


    def test_deflection_from_kappa(self):
        sis = SIS()
        deltaPix = 0.01
        x_grid, y_grid = util.make_grid(numPix=1000, deltapix=deltaPix)
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        f_x, f_y = sis.derivatives(x_grid, y_grid, **kwargs_sis)
        f_x = util.array2image(f_x)
        kappa = util.array2image((f_xx + f_yy) / 2.)
        f_x_num, f_y_num = convergence_integrals.deflection_from_kappa_grid(kappa, deltaPix)

        x1, y1 = 550, 500
        # test relative potential at two different point way inside the kappa map
        npt.assert_almost_equal(f_x[x1, y1], f_x_num[x1, y1], decimal=2)

    def test_deflection_from_kappa_adaptiv(self):
        sis = SIS()
        deltaPix = 0.01
        kwargs_sis = {'theta_E': 1., 'center_x': 0, 'center_y': 0}
        low_res_factor = 5
        high_res_kernel_size = 5
        x_grid, y_grid = util.make_grid(numPix=1000, deltapix=deltaPix)

        f_xx, f_yy, _ = sis.hessian(x_grid, y_grid, **kwargs_sis)
        kappa = util.array2image((f_xx + f_yy) / 2.)
        f_x_num, f_y_num = convergence_integrals.deflection_from_kappa_grid_adaptive(kappa, deltaPix, low_res_factor, high_res_kernel_size)

        x_grid_low, y_grid_low = util.make_grid(numPix=1000/low_res_factor, deltapix=deltaPix*low_res_factor)
        f_x_low, f_y_low = sis.derivatives(x_grid_low, y_grid_low, **kwargs_sis)
        f_x_low = util.array2image(f_x_low)
        f_y_low = util.array2image(f_y_low)
        x1, y1 = 50, 51
        # test relative potential at two different point way inside the kappa map
        npt.assert_almost_equal(f_x_low[x1, y1], f_x_num[x1, y1], decimal=2)
        print(f_x_low[x1, y1], f_x_num[x1, y1])
        npt.assert_almost_equal(f_y_low[x1, y1], f_y_num[x1, y1], decimal=2)

    def test_sersic(self):
        from lenstronomy.LensModel.Profiles.sersic import Sersic
        from lenstronomy.LightModel.Profiles.sersic import Sersic as SersicLight
        sersic_lens = Sersic()
        sersic_light = SersicLight()
        kwargs_light = {'n_sersic': 2, 'R_sersic': 0.5, 'I0_sersic': 1, 'center_x': 0, 'center_y': 0}
        kwargs_lens = {'n_sersic': 2, 'R_sersic': 0.5, 'k_eff': 1, 'center_x': 0, 'center_y': 0}
        deltaPix = 0.01
        numPix = 1000
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltaPix)
        x_grid2d = util.array2image(x_grid)
        y_grid2d = util.array2image(y_grid)

        f_xx, f_yy, _ = sersic_lens.hessian(x_grid, y_grid, **kwargs_lens)
        f_x, f_y = sersic_lens.derivatives(x_grid, y_grid, **kwargs_lens)
        f_x = util.array2image(f_x)
        kappa = util.array2image((f_xx + f_yy) / 2.)
        f_x_num, f_y_num = convergence_integrals.deflection_from_kappa_grid(kappa, deltaPix)
        x1, y1 = 500, 550
        x0, y0 = int(numPix/2.), int(numPix/2.)
        npt.assert_almost_equal(f_x[x1, y1], f_x_num[x1, y1], decimal=2)
        f_num = convergence_integrals.potential_from_kappa_grid(kappa, deltaPix)
        f_ = sersic_lens.function(x_grid2d[x1, y1], y_grid2d[x1, y1], **kwargs_lens)
        f_00 = sersic_lens.function(x_grid2d[x0, y0], y_grid2d[x0, y0], **kwargs_lens)
        npt.assert_almost_equal(f_ - f_00, f_num[x1, y1] - f_num[x0, y0], decimal=2)


if __name__ == '__main__':
    pytest.main()
