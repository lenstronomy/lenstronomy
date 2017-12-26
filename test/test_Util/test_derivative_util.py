__author__ = 'sibirrer'

import lenstronomy.Util.derivative_util as calc_util
import pytest


class TestCalcUtil(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        pass


    def test_d_r_dx(self):
        x = 1
        y = 0
        out = calc_util.d_r_dx(x, y)
        assert out == 1

    def test_d_r_dy(self):
        x = 1
        y = 0
        out = calc_util.d_r_dy(x, y)
        assert out == 0

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


if __name__ == '__main__':
    pytest.main()