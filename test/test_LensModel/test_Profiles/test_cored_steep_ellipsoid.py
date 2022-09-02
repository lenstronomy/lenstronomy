__author__ = 'sibirrer'


import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Util import param_util


class TestCSP(object):
    """
    tests the cored steep ellipsoid (CSE)
    """
    def setup(self):
        from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSE
        self.CSP = CSE(axis='product_avg')

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

    def test_ellipticity(self):
        """
        test the definition of the ellipticity normalization (along major axis or product averaged axes)
        """
        x, y = np.linspace(start=0.001, stop=10, num=100), np.zeros(100)
        kwargs_round = {'a': 2, 's': 1, 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0}
        phi_q, q = param_util.ellipticity2phi_q(0.3, 0)
        kwargs = {'a': 2, 's': 1, 'e1': 0.3, 'e2': 0., 'center_x': 0, 'center_y': 0}

        f_xx, f_xy, f_yx, f_yy = self.CSP.hessian(x, y, **kwargs_round)
        kappa_round = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.CSP.hessian(x, y, **kwargs)
        kappa_major = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.CSP.hessian(y, x, **kwargs)
        kappa_minor = 1. / 2 * (f_xx + f_yy)

        # import matplotlib.pyplot as plt
        # plt.plot(x, kappa_major/kappa_round, ',-', label='major/round', alpha=0.5)
        # plt.plot(x, kappa_minor/kappa_round, '--', label='minor/round', alpha=0.5)
        #
        # plt.plot(x, np.sqrt(kappa_minor*kappa_major)/kappa_round,label='prod/kappa_round')
        # plt.legend()
        # plt.show()

        npt.assert_almost_equal(kappa_round,np.sqrt(kappa_minor*kappa_major), decimal=1)

if __name__ == '__main__':
    pytest.main()
