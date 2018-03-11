__author__ = 'sibirrer'


import numpy as np
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
            self.sie = SIE()
            self.spemd = SPEMD()

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


if __name__ == '__main__':
    pytest.main()

