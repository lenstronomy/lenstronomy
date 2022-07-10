__author__ = 'sibirrer'


import numpy as np
import pytest
import numpy.testing as npt


class TestCSP(object):
    """
    tests the cored steep ellipsoid (CSE)
    """
    def setup(self):
        from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSE
        self.CSP = CSE()

    def test_function(self):

        kwargs = {'a': 2, 's': 1, 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0}

        x = np.array([1., 2])
        y = np.array([2, 0])
        f_ = self.CSP.function(x, y, **kwargs)
        npt.assert_almost_equal(f_, [1.09016, 0.96242], decimal=5)

    def test_derivatives(self):
        kwargs = {'a': 2, 's': 1, 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0}

        x = np.array([1., 2])
        y = np.array([2, 0])
        f_x, f_y = self.CSP.derivatives(x, y, **kwargs)
        npt.assert_almost_equal(f_x, [0.2367, 0.55279], decimal=5)
        npt.assert_almost_equal(f_y, [0.4734, 0.], decimal=5)

    def test_hessian(self):
        kwargs = {'a': 2, 's': 1, 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0}

        x = np.array([1., 2])
        y = np.array([2, 0])
        f_xx, f_xy, f_yx, f_yy = self.CSP.hessian(x, y, **kwargs)
        npt.assert_almost_equal(f_xy, f_yx, decimal=5)
        npt.assert_almost_equal(f_xx, [0.16924, -0.09751], decimal=5)
        npt.assert_almost_equal(f_xy, [-0.13493, -0.], decimal=5)
        npt.assert_almost_equal(f_yy, [-0.03315,  0.27639], decimal=5)


if __name__ == '__main__':
    pytest.main()
