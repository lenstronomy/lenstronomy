__author__ = 'sibirrer'
import pytest
import numpy as np

import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.interpol import Interpol_func

class TestInterpol(object):

    def test_do_interpol(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix,deltaPix)
        sis = SIS()
        kwargs_SIS = {'theta_E':1., 'center_x':0.5, 'center_y':-0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_yy_sis, f_xy_sis = sis.hessian(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol_func()
        interp_func_loop = Interpol_func(grid=False)
        interp_func.do_interp(x_axes, y_axes, util.array2image(f_sis), util.array2image(f_x_sis), util.array2image(f_y_sis), util.array2image(f_xx_sis), util.array2image(f_yy_sis), util.array2image(f_xy_sis))
        interp_func_loop.do_interp(x_axes, y_axes, util.array2image(f_sis), util.array2image(f_x_sis), util.array2image(f_y_sis), util.array2image(f_xx_sis), util.array2image(f_yy_sis), util.array2image(f_xy_sis))

        # test derivatives
        assert interp_func.derivatives(1,0) == sis.derivatives(1,0, **kwargs_SIS)
        assert interp_func.derivatives(1, 0) == interp_func_loop.derivatives(1, 0)
        alpha1_interp, alpha2_interp = interp_func.derivatives(np.array([0,1,0,1]), np.array([1,1,2,2]))
        alpha1_interp_loop, alpha2_interp_loop = interp_func_loop.derivatives(np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]))
        alpha1_true, alpha2_true = sis.derivatives(np.array([0,1,0,1]),np.array([1,1,2,2]), **kwargs_SIS)
        assert alpha1_interp[0] == alpha1_true[0]
        assert alpha1_interp[1] == alpha1_true[1]
        assert alpha1_interp[0] == alpha1_interp_loop[0]
        assert alpha1_interp[1] == alpha1_interp_loop[1]
        # test hessian
        assert interp_func.hessian(1,0) == sis.hessian(1,0, **kwargs_SIS)
        f_xx_interp, f_yy_interp, f_xy_interp = interp_func.hessian(np.array([0,1,0,1]), np.array([1,1,2,2]))
        f_xx_interp_loop, f_yy_interp_loop, f_xy_interp_loop = interp_func_loop.hessian(np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]))
        f_xx_true, f_yy_true, f_xy_true = sis.hessian(np.array([0,1,0,1]),np.array([1,1,2,2]), **kwargs_SIS)
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
        x_grid_interp, y_grid_interp = util.make_grid(numPix,deltaPix)
        sis = SIS()
        kwargs_SIS = {'theta_E':1., 'center_x': 0.5, 'center_y': -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_yy_sis, f_xy_sis = sis.hessian(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol_func()
        interp_func.do_interp(x_axes, y_axes, util.array2image(f_sis), util.array2image(f_x_sis), util.array2image(f_y_sis), util.array2image(f_xx_sis), util.array2image(f_yy_sis), util.array2image(f_xy_sis))
        x, y = 1., 1.
        alpha_x, alpha_y = interp_func.derivatives(x, y, **{})
        assert alpha_x == 0.31622776601683794

    def test_kwargs_interpolation(self):
        numPix = 101
        deltaPix = 0.1
        x_grid_interp, y_grid_interp = util.make_grid(numPix,deltaPix)
        sis = SIS()
        kwargs_SIS = {'theta_E':1., 'center_x': 0.5, 'center_y': -0.5}
        f_sis = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x_sis, f_y_sis = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx_sis, f_yy_sis, f_xy_sis = sis.hessian(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        interp_func = Interpol_func()
        kwargs_interp = {'grid_interp_x': x_axes, 'grid_interp_y': y_axes, 'f_': util.array2image(f_sis), 'f_x': util.array2image(f_x_sis), 'f_y': util.array2image(f_y_sis), 'f_xx': util.array2image(f_xx_sis), 'f_yy': util.array2image(f_yy_sis), 'f_xy': util.array2image(f_xy_sis)}
        x, y = 1., 1.
        alpha_x, alpha_y = interp_func.derivatives(x, y, **kwargs_interp)
        assert alpha_x == 0.31622776601683794


if __name__ == '__main__':
    pytest.main("-k TestSourceModel")