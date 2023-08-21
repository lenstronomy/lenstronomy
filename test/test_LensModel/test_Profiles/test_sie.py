__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
import lenstronomy.Util.param_util as param_util


class TestSIE(object):
        """Tests the Gaussian methods."""
        def setup_method(self):
            from lenstronomy.LensModel.Profiles.sie import SIE
            from lenstronomy.LensModel.Profiles.epl import EPL
            from lenstronomy.LensModel.Profiles.nie import NIE
            self.sie = SIE(NIE=False)
            self.sie_nie = SIE(NIE=True)
            self.epl = EPL()
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
            values_spemd = self.epl.function(x, y, theta_E, gamma, e1, e2)
            assert values == values_spemd

            values_nie = self.sie_nie.function(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.function(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values_nie, values_spemd, decimal=6)

        def test_derivatives(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.7
            phi_G = 1.
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.sie.derivatives(x, y, theta_E, e1, e2)
            gamma = 2
            values_spemd = self.epl.derivatives(x, y, theta_E, gamma, e1, e2)
            assert values == values_spemd

            values = self.sie_nie.derivatives(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values, values_spemd, decimal=6)

        def test_hessian(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.7
            phi_G = 1.
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.sie.hessian(x, y, theta_E, e1, e2)
            gamma = 2
            values_spemd = self.epl.hessian(x, y, theta_E, gamma, e1, e2)
            assert values[0] == values_spemd[0]

            values = self.sie_nie.hessian(x, y, theta_E, e1, e2)
            s_scale = 0.0000001
            values_spemd = self.nie.hessian(x, y, theta_E, e1, e2, s_scale)
            npt.assert_almost_equal(values, values_spemd, decimal=5)


if __name__ == '__main__':
    pytest.main()
