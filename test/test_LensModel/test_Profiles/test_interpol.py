__author__ = "sibirrer"
import pytest
import numpy as np
import numpy.testing as npt

import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.interpol import Interpol, InterpolScaled


class TestInterpol(object):
    def setup_method(self):
        pass

    def test_do_interpol(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()
        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = sis.hessian(
            x_grid_interp, y_grid_interp, **kwargs_SIS
        )
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol(grid=True)
        interp_func_loop = Interpol(grid=False)
        interp_func.do_interp(
            x_axes,
            y_axes,
            util.array2image(f_sis),
            util.array2image(f_x_sis),
            util.array2image(f_y_sis),
            util.array2image(f_xx_sis),
            util.array2image(f_yy_sis),
            util.array2image(f_xy_sis),
        )
        interp_func_loop.do_interp(
            x_axes,
            y_axes,
            util.array2image(f_sis),
            util.array2image(f_x_sis),
            util.array2image(f_y_sis),
            util.array2image(f_xx_sis),
            util.array2image(f_yy_sis),
            util.array2image(f_xy_sis),
        )

        # test derivatives
        print(interp_func.derivatives(0, 1))
        print(sis.derivatives(1, 0, **kwargs_SIS))
        # assert interp_func.derivatives(1, 0) == sis.derivatives(1, 0, **kwargs_SIS)
        assert interp_func.derivatives(1, 0) == interp_func_loop.derivatives(1, 0)
        alpha1_interp, alpha2_interp = interp_func.derivatives(
            np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2])
        )
        alpha1_interp_loop, alpha2_interp_loop = interp_func_loop.derivatives(
            np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2])
        )
        alpha1_true, alpha2_true = sis.derivatives(
            np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]), **kwargs_SIS
        )
        assert alpha1_interp[0] == alpha1_true[0]
        assert alpha1_interp[1] == alpha1_true[1]
        assert alpha1_interp[0] == alpha1_interp_loop[0]
        assert alpha1_interp[1] == alpha1_interp_loop[1]
        # test hessian
        assert interp_func.hessian(1, 0) == sis.hessian(1, 0, **kwargs_SIS)
        f_xx_interp, f_xy_interp, f_yx_interp, f_yy_interp = interp_func.hessian(
            np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2])
        )
        (
            f_xx_interp_loop,
            f_xy_interp_loop,
            f_yx_interp_loop,
            f_yy_interp_loop,
        ) = interp_func_loop.hessian(np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]))
        f_xx_true, f_xy_true, f_yx_true, f_yy_true = sis.hessian(
            np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]), **kwargs_SIS
        )
        assert f_xx_interp[0] == f_xx_true[0]
        assert f_xx_interp[1] == f_xx_true[1]
        assert f_xy_interp[0] == f_xy_true[0]
        assert f_xy_interp[1] == f_xy_true[1]
        assert f_xx_interp[0] == f_xx_interp_loop[0]
        assert f_xx_interp[1] == f_xx_interp_loop[1]
        assert f_xy_interp[0] == f_xy_interp_loop[0]
        assert f_xy_interp[1] == f_xy_interp_loop[1]
        # test all

    def test_call(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()
        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = sis.hessian(
            x_grid_interp, y_grid_interp, **kwargs_SIS
        )
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol(grid=True)
        interp_func.do_interp(
            x_axes,
            y_axes,
            util.array2image(f_sis),
            util.array2image(f_x_sis),
            util.array2image(f_y_sis),
            util.array2image(f_xx_sis),
            util.array2image(f_yy_sis),
            util.array2image(f_xy_sis),
        )
        x, y = 1.0, 1.0
        alpha_x, alpha_y = interp_func.derivatives(x, y, **{})
        assert alpha_x == 0.31622776601683794

    def test_kwargs_interpolation(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()
        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = sis.hessian(
            x_grid_interp, y_grid_interp, **kwargs_SIS
        )
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol()
        kwargs_interp = {
            "grid_interp_x": x_axes,
            "grid_interp_y": y_axes,
            "f_": util.array2image(f_sis),
            "f_x": util.array2image(f_x_sis),
            "f_y": util.array2image(f_y_sis),
            "f_xx": util.array2image(f_xx_sis),
            "f_yy": util.array2image(f_yy_sis),
            "f_xy": util.array2image(f_xy_sis),
        }
        x, y = 1.0, 1.0
        alpha_x, alpha_y = interp_func.derivatives(x, y, **kwargs_interp)
        assert alpha_x == 0.31622776601683794

        x, y = 1.0, 0.0
        alpha_x, alpha_y = interp_func.derivatives(x, y, **kwargs_interp)
        alpha_x_true, alpha_y_true = sis.derivatives(x, y, **kwargs_SIS)
        npt.assert_almost_equal(alpha_x, alpha_x_true, decimal=10)
        npt.assert_almost_equal(alpha_y, alpha_y_true, decimal=10)

        f_ = interp_func.function(x, y, **kwargs_interp)
        f_true = sis.derivatives(x, y, **kwargs_SIS)
        npt.assert_almost_equal(f_, f_true, decimal=10)

    def test_hessian_finite_differential(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()
        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol()
        kwargs_interp = {
            "grid_interp_x": x_axes,
            "grid_interp_y": y_axes,
            "f_": util.array2image(f_sis),
            "f_x": util.array2image(f_x_sis),
            "f_y": util.array2image(f_y_sis),
        }
        x, y = 1.0, 0.0
        f_xx, f_xy, f_yx, f_yy = interp_func.hessian(x, y, **kwargs_interp)
        f_xx_true, f_xy_true, f_yx_true, f_yy_true = sis.hessian(x, y, **kwargs_SIS)
        npt.assert_almost_equal(f_xx, f_xx_true, decimal=1)
        npt.assert_almost_equal(f_xy, f_xy_true, decimal=1)
        npt.assert_almost_equal(f_yx, f_yx_true, decimal=1)
        npt.assert_almost_equal(f_yy, f_yy_true, decimal=1)

    def test_interp_func_scaled(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()
        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = sis.hessian(
            x_grid_interp, y_grid_interp, **kwargs_SIS
        )
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        kwargs_interp = {
            "grid_interp_x": x_axes,
            "grid_interp_y": y_axes,
            "f_": util.array2image(f_sis),
            "f_x": util.array2image(f_x_sis),
            "f_y": util.array2image(f_y_sis),
            "f_xx": util.array2image(f_xx_sis),
            "f_yy": util.array2image(f_yy_sis),
            "f_xy": util.array2image(f_xy_sis),
        }
        interp_func = InterpolScaled(grid=False)
        x, y = 1.0, 1.0
        alpha_x, alpha_y = interp_func.derivatives(
            x, y, scale_factor=1, **kwargs_interp
        )
        assert alpha_x == 0.31622776601683794

        f_ = interp_func.function(x, y, scale_factor=1.0, **kwargs_interp)
        npt.assert_almost_equal(f_, 1.5811388300841898)

        f_xx, f_xy, f_yx, f_yy = interp_func.hessian(
            x, y, scale_factor=1.0, **kwargs_interp
        )
        npt.assert_almost_equal(f_xx, 0.56920997883030822, decimal=8)
        npt.assert_almost_equal(f_yy, 0.063245553203367583, decimal=8)
        npt.assert_almost_equal(f_xy, -0.18973665961010275, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

        x_grid, y_grid = util.make_grid(10, deltaPix)
        f_xx, f_xy, f_yx, f_yy = interp_func.hessian(
            x_grid, y_grid, scale_factor=1.0, **kwargs_interp
        )
        npt.assert_almost_equal(f_xx[0], 0, decimal=2)

    def test_shift(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        sis = SIS()

        kwargs_SIS = {"theta_E": 1.0, "center_x": 0.5, "center_y": -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = sis.hessian(
            x_grid_interp, y_grid_interp, **kwargs_SIS
        )
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        kwargs_interp = {
            "grid_interp_x": x_axes,
            "grid_interp_y": y_axes,
            "f_": util.array2image(f_sis),
            "f_x": util.array2image(f_x_sis),
            "f_y": util.array2image(f_y_sis),
            "f_xx": util.array2image(f_xx_sis),
            "f_yy": util.array2image(f_yy_sis),
            "f_xy": util.array2image(f_xy_sis),
        }
        interp_func = Interpol(grid=False)
        x, y = 1.0, 1.0
        alpha_x, alpha_y = interp_func.derivatives(x, y, **kwargs_interp)
        assert alpha_x == 0.31622776601683794

        interp_func = Interpol(grid=False)
        x_shift = 1.0
        kwargs_shift = {
            "grid_interp_x": x_axes + x_shift,
            "grid_interp_y": y_axes,
            "f_": util.array2image(f_sis),
            "f_x": util.array2image(f_x_sis),
            "f_y": util.array2image(f_y_sis),
            "f_xx": util.array2image(f_xx_sis),
            "f_yy": util.array2image(f_yy_sis),
            "f_xy": util.array2image(f_xy_sis),
        }
        alpha_x_shift, alpha_y_shift = interp_func.derivatives(
            x + x_shift, y, **kwargs_shift
        )
        npt.assert_almost_equal(alpha_x_shift, alpha_x, decimal=10)


if __name__ == "__main__":
    pytest.main()
