__author__ = 'sibirrer'

import lenstronomy.Util.derivative_util as calc_util
import pytest
from lenstronomy.Util import util
from lenstronomy.Util import param_util
import numpy.testing as npt
import numpy as np


class TestCalcUtil(object):
    """Tests the Gaussian methods."""
    def setup_method(self):
        pass

    def test_d_r_dx(self):
        x = 1
        y = 0
        out = calc_util.d_r_dx(x, y)
        assert out == 1
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        dx = 0.000001
        out = calc_util.d_r_dx(x, y)
        r, phi = param_util.cart2polar(x, y)
        r_dx, phi_dx = param_util.cart2polar(x + dx, y)
        dr_dx = (r_dx - r) / dx
        npt.assert_almost_equal(dr_dx, out, decimal=5)

    def test_d_r_dy(self):
        x = 1
        y = 0
        out = calc_util.d_r_dy(x, y)
        assert out == 0
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        dy = 0.000001
        out = calc_util.d_r_dy(x, y)
        r, phi = param_util.cart2polar(x, y)
        r_dy, phi_dy = param_util.cart2polar(x, y + dy)
        dr_dy = (r_dy - r) / dy
        npt.assert_almost_equal(dr_dy, out, decimal=5)

    def test_d_x_diffr_dx(self):
        x = 1
        y = 0
        out = calc_util.d_x_diffr_dx(x, y)
        assert out == 0
        x = 0
        y = 1
        out = calc_util.d_x_diffr_dx(x, y)
        assert out == 1

    def test_d_y_diffr_dx(self):
        x = 1
        y = 0
        out = calc_util.d_y_diffr_dx(x, y)
        assert out == 0

        x = 0
        y = 1
        out = calc_util.d_y_diffr_dx(x, y)
        assert out == 0

    def test_d_y_diffr_dy(self):
        x = 1
        y = 0
        out = calc_util.d_y_diffr_dy(x, y)
        assert out == 1

        x = 0
        y = 1
        out = calc_util.d_y_diffr_dy(x, y)
        assert out == 0

    def test_d_x_diffr_dy(self):
        x = 1
        y = 0
        out = calc_util.d_x_diffr_dy(x, y)
        assert out == 0

        x = 0
        y = 1
        out = calc_util.d_x_diffr_dy(x, y)
        assert out == 0

    def test_d_phi_dx(self):
        x, y = np.array([1., 0., -1.]), np.array([1., 1., -1.])
        dx, dy = 0.0001, 0.0001
        r, phi = param_util.cart2polar(x, y, center_x=0, center_y=0)

        d_phi_dx = calc_util.d_phi_dx(x, y)
        d_phi_dy = calc_util.d_phi_dy(x, y)
        r_dx, phi_dx = param_util.cart2polar(x + dx, y, center_x=0, center_y=0)
        r_dy, phi_dy = param_util.cart2polar(x, y + dy, center_x=0, center_y=0)
        d_phi_dx_num = (phi_dx - phi) / dx
        d_phi_dy_num = (phi_dy - phi) / dy
        npt.assert_almost_equal(d_phi_dx, d_phi_dx_num, decimal=4)
        npt.assert_almost_equal(d_phi_dy, d_phi_dy_num, decimal=4)

    def test_d_phi_dxx(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        delta = 0.00001
        d_phi_dx = calc_util.d_phi_dx(x, y)
        d_phi_dx_delta = calc_util.d_phi_dx(x + delta, y)
        d_phi_dy = calc_util.d_phi_dy(x, y)
        d_phi_dxx = calc_util.d_phi_dxx(x, y)
        d_phi_dxx_num = (d_phi_dx_delta - d_phi_dx) / delta
        npt.assert_almost_equal(d_phi_dxx_num, d_phi_dxx, decimal=1)

        d_phi_dy_delta = calc_util.d_phi_dy(x, y + delta)
        d_phi_dyy = calc_util.d_phi_dyy(x, y)
        d_phi_dyy_num = (d_phi_dy_delta - d_phi_dy) / delta
        npt.assert_almost_equal(d_phi_dyy_num, d_phi_dyy, decimal=1)

        d_phi_dx_delta_y = calc_util.d_phi_dx(x, y + delta)
        d_phi_dxy = calc_util.d_phi_dxy(x, y)
        d_phi_dxy_num = (d_phi_dx_delta_y - d_phi_dx) / delta
        npt.assert_almost_equal(d_phi_dxy_num, d_phi_dxy, decimal=1)

    def test_d_r_dxx(self):
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        delta = 0.00001
        d_r_dx = calc_util.d_r_dx(x, y)
        d_r_dx_delta = calc_util.d_r_dx(x + delta, y)
        d_r_dy = calc_util.d_r_dy(x, y)
        d_r_dxx = calc_util.d_r_dxx(x, y)
        d_r_dxx_num = (d_r_dx_delta - d_r_dx) / delta
        npt.assert_almost_equal(d_r_dxx_num, d_r_dxx, decimal=1)

        d_r_dy_delta = calc_util.d_r_dy(x, y + delta)
        d_r_dyy = calc_util.d_r_dyy(x, y)
        d_r_dyy_num = (d_r_dy_delta - d_r_dy) / delta
        npt.assert_almost_equal(d_r_dyy_num, d_r_dyy, decimal=1)

        d_r_dx_delta_y = calc_util.d_r_dx(x, y + delta)
        d_r_dxy = calc_util.d_r_dxy(x, y)
        d_r_dxy_num = (d_r_dx_delta_y - d_r_dx) / delta
        npt.assert_almost_equal(d_r_dxy_num, d_r_dxy, decimal=1)


if __name__ == '__main__':
    pytest.main()
