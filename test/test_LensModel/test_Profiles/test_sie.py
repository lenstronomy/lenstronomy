__author__ = 'sibirrer'


import numpy as np
import numpy.testing as npt
import pytest
import lenstronomy.Util.param_util as param_util

try:
    import fastell4py
except:
    print("Warning: fastell4py not available, tests will be trivially fulfilled without giving the right answer!")


class TestSIE(object):
        """
        tests the Gaussian methods
        """
        def setup(self):
            from lenstronomy.LensModel.Profiles.sie import SIE
            from lenstronomy.LensModel.Profiles.spemd import SPEMD
            from lenstronomy.LensModel.Profiles.nie import NIE
            self.sie = SIE()
            self.sie_nie = SIE(NIE=True)
            self.spemd = SPEMD()
            self.nie = NIE()

        def test_function(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.9
            phi_G = 1.
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.sie.function(x, y, theta_E, e1, e2)
            gamma = 2
            values_spemd = self.spemd.function(x, y, theta_E, gamma, e1, e2)
            assert values == values_spemd

            values = self.sie_nie.function(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.function(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values, values_spemd, decimal=6)

        def test_derivatives(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.9
            phi_G = 1.
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.sie.derivatives(x, y, theta_E, e1, e2)
            gamma = 2
            values_spemd = self.spemd.derivatives(x, y, theta_E, gamma, e1, e2)
            assert values == values_spemd

            values = self.sie_nie.derivatives(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values, values_spemd, decimal=6)

        def test_hessian(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.9
            phi_G = 1.
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.sie.hessian(x, y, theta_E, e1, e2)
            gamma = 2
            values_spemd = self.spemd.hessian(x, y, theta_E, gamma, e1, e2)
            assert values[0] == values_spemd[0]

            values = self.sie_nie.hessian(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.hessian(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values, values_spemd, decimal=5)


if __name__ == '__main__':
    pytest.main()
