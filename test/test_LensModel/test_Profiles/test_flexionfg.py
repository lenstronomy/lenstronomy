__author__ = 'ylilan'


from lenstronomy.LensModel.Profiles.flexionfg import Flexionfg
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest

class TestFlexionfg(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.flex = Flexionfg()
        F1, F2, G1, G2= 0.02, 0.03, -0.04, -0.05
        self.kwargs_lens = {'F1': F1, 'F2': F2, 'G1': G1, 'G2': G2}

    def test_transform_fg(self):
        values=self.flex.transform_fg(**self.kwargs_lens)
        g1,g2,g3,g4 = 0.01, 0.02, 0.03, 0.04
        npt.assert_almost_equal(values[0], g1, decimal=5)
        npt.assert_almost_equal(values[1], g2, decimal=5)
        npt.assert_almost_equal(values[2], g3, decimal=5)
        npt.assert_almost_equal(values[3], g4, decimal=5)

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.135, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0, decimal=5)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.09, decimal=5)
        npt.assert_almost_equal(values[1], 0.18666666666666668, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.flex.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0], 0.105, decimal=5)
        npt.assert_almost_equal(f_y[0], 0.15, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.flex.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.105, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.15, decimal=5)

    def test_hessian(self):
        x = np.array(1)
        y = np.array(2)
        f_xx, f_yy, f_xy = self.flex.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0.05, decimal=5)
        npt.assert_almost_equal(f_yy, 0.11, decimal=5)
        npt.assert_almost_equal(f_xy, 0.08, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.flex.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.05, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.11, decimal=5)
        npt.assert_almost_equal(values[2][0], 0.08, decimal=5)

    def test_flexion(self):
        x = np.array(0)
        y = np.array(2)
        flex = LensModel(['FLEXIONFG'])
        f_xxx, f_xxy, f_xyy, f_yyy = flex.flexion(x, y, [self.kwargs_lens])
        _g1,_g2,_g3,_g4 = self.flex.transform_fg(**self.kwargs_lens)
        npt.assert_almost_equal(f_xxx, _g1, decimal=9)
        npt.assert_almost_equal(f_xxy, _g2, decimal=9)
        npt.assert_almost_equal(f_xyy, _g3, decimal=9)
        npt.assert_almost_equal(f_yyy, _g4, decimal=9)

    def test_magnification(self):
        ra_0, dec_0 = 1, -1

        flex = LensModel(['FLEXIONFG'])
        F1, F2, G1, G2 = 0.02, 0.03, -0.04, -0.05
        kwargs = {'F1': F1, 'F2': F2, 'G1': G1, 'G2': G2, 'ra_0': ra_0, 'dec_0': dec_0}
        mag = flex.magnification(ra_0, dec_0, [kwargs])
        npt.assert_almost_equal(mag, 1, decimal=8)


if __name__ == '__main__':
    pytest.main()