from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.radial_interpolated import RadialInterpolate
from lenstronomy.Util import util
import numpy as np
import numpy.testing as npt


class TestRadialInterpolation(object):
    """
    testing radial interpolation class
    test case is a SIS profile and a constant mass sheet

    """
    def setup_method(self):
        self.rad_interp = RadialInterpolate()

    def test_sis(self):
        interp_profile = LensModel(lens_model_list=["RADIAL_INTERPOL"])
        sis = LensModel(lens_model_list=['SIS'])
        kwargs_sis = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        r_bin_log = np.logspace(-4, 1, 200)
        kappa_r_sis = sis.kappa(r_bin_log, 0, kwargs=kwargs_sis)
        kwargs_interp = [{"r_bin": r_bin_log, "kappa_r": kappa_r_sis}]

        x, y = util.make_grid(numPix=10, deltapix=0.1)

        # hessian
        f_xx_int, f_xy_int, f_yx_int, f_yy_int = interp_profile.hessian(x, y, kwargs_interp)
        f_xx, f_xy, f_yx, f_yy = sis.hessian(x, y, kwargs_sis)
        kappa_int = 1/2 * (f_xx_int + f_yy_int)
        kappa = 1/2 * (f_xx + f_yy)

        gamma1 = 1/2 * (f_xx - f_yy)
        gamma1_int = 1 / 2 * (f_xx_int - f_yy_int)

        npt.assert_almost_equal(kappa_int / kappa, 1, decimal=3)
        npt.assert_almost_equal(f_xx_int, f_xx, decimal=2)
        npt.assert_almost_equal(f_xy_int, f_xy, decimal=2)
        npt.assert_almost_equal(f_yx_int, f_yx, decimal=2)
        npt.assert_almost_equal(f_yy_int, f_yy, decimal=2)

        # deflection
        f_x_int, f_y_int = interp_profile.alpha(x, y, kwargs_interp)
        f_x, f_y = sis.alpha(x, y, kwargs_sis)
        npt.assert_almost_equal(f_x_int, f_x, decimal=3)
        npt.assert_almost_equal(f_y_int, f_y, decimal=3)

        # potential
        f_int = interp_profile.potential(x, y, kwargs_interp)
        f_ = sis.potential(x, y, kwargs_sis)
        npt.assert_almost_equal(f_int, f_, decimal=3)

    def test_mass_sheet(self):
        interp_profile = LensModel(lens_model_list=["RADIAL_INTERPOL"])
        mass_sheet = LensModel(lens_model_list=['CONVERGENCE'])
        kwargs_convergence = [{"kappa": 0.5}]
        r_bin = np.linspace(start=0, stop=1, num=100)
        kappa_r_sis = mass_sheet.kappa(r_bin, 0, kwargs=kwargs_convergence)
        kwargs_interp = [{"r_bin": r_bin, "kappa_r": kappa_r_sis}]

        x, y = util.make_grid(numPix=10, deltapix=0.1)

        # hessian
        f_xx_int, f_xy_int, f_yx_int, f_yy_int = interp_profile.hessian(x, y, kwargs_interp)
        f_xx, f_xy, f_yx, f_yy = mass_sheet.hessian(x, y, kwargs_convergence)
        kappa_int = 1 / 2 * (f_xx_int + f_yy_int)
        kappa = 1 / 2 * (f_xx + f_yy)
        npt.assert_almost_equal(kappa_int / kappa, 1, decimal=3)

        npt.assert_almost_equal(f_xx_int, f_xx, decimal=3)
        npt.assert_almost_equal(f_xy_int, f_xy, decimal=3)
        npt.assert_almost_equal(f_yx_int, f_yx, decimal=3)
        npt.assert_almost_equal(f_yy_int, f_yy, decimal=3)

        # deflection
        f_x_int, f_y_int = interp_profile.alpha(x, y, kwargs_interp)
        f_x, f_y = mass_sheet.alpha(x, y, kwargs_convergence)
        npt.assert_almost_equal(f_x_int, f_x, decimal=3)
        npt.assert_almost_equal(f_y_int, f_y, decimal=3)

        # potential
        f_int = interp_profile.potential(x, y, kwargs_interp)
        f_ = mass_sheet.potential(x, y, kwargs_convergence)
        npt.assert_almost_equal(f_int, f_, decimal=3)
