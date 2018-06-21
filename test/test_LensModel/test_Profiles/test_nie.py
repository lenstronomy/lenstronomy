__author__ = 'sibirrer'


import numpy as np
import numpy.testing as npt
import pytest
import lenstronomy.Util.param_util as param_util

try:
    import fastell4py
    bool_test = True
except:
    bool_test = False
    print("Warning: fastell4py not available, tests will not crosscheck with fastell4py on your machine")


class TestNIE(object):
        """
        tests the Gaussian methods
        """
        def setup(self):
            from lenstronomy.LensModel.Profiles.nie import NIE
            from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
            from lenstronomy.LensModel.Profiles.sis import SIS
            self.nie = NIE()
            self.spemd = SPEMD_SMOOTH()
            self.sis = SIS()

        def test_function(self):
            y = np.array([1., 2])
            x = np.array([0., 0.])
            theta_E = 1.
            q = 0.9999
            s = 0.00001
            phi_G = 0
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

            values = self.nie.function(x, y, theta_E, e1, e2, s_scale=s)
            delta_pot = values[1] - values[0]
            values_spemd = self.sis.function(x, y, theta_E)
            delta_pot_spemd = values_spemd[1] - values_spemd[0]
            npt.assert_almost_equal(delta_pot, delta_pot_spemd, decimal=4)
            if bool_test is True:
                q = 0.99
                s = 0.000001
                phi_G = 0
                e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
                values = self.nie.function(x, y, theta_E, e1, e2, s_scale=s)
                delta_pot = values[1] - values[0]
                gamma = 2.
                values_spemd = self.spemd.function(x, y, theta_E, gamma, e1, e2, s_scale=s)
                delta_pot_spemd = values_spemd[1] - values_spemd[0]
                npt.assert_almost_equal(delta_pot, delta_pot_spemd, decimal=2)

        def test_derivatives(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.99999
            phi_G = 0
            s = 0.0000001
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            f_x, f_y = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale=s)
            f_x_spemd, f_y_spemd = self.sis.derivatives(x, y, theta_E)
            npt.assert_almost_equal(f_x, f_x_spemd, decimal=4)
            npt.assert_almost_equal(f_y, f_y_spemd, decimal=4)
            if bool_test is True:
                q = 0.99
                s = 0.000001
                phi_G = 0
                e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
                f_x, f_y = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale=s)
                gamma = 2.
                f_x_spemd, f_y_spemd = self.spemd.derivatives(x, y, theta_E, gamma, e1, e2, s_scale=s)
                print(f_x/f_x_spemd, 'ratio deflections')
                print(1+(1-q)/2)
                npt.assert_almost_equal(f_x, f_x_spemd, decimal=2)
                npt.assert_almost_equal(f_y, f_y_spemd, decimal=2)


        def test_hessian(self):
            x = np.array([1])
            y = np.array([2])
            theta_E = 1.
            q = 0.999999
            phi_G = 0
            s = 0.0000001
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            f_xx, f_yy, f_xy = self.nie.hessian(x, y, theta_E, e1, e2, s_scale=s)
            f_xx_spemd, f_yy_spemd, f_xy_spemd = self.sis.hessian(x, y, theta_E)
            npt.assert_almost_equal(f_xx, f_xx_spemd, decimal=4)
            npt.assert_almost_equal(f_yy, f_yy_spemd, decimal=4)
            npt.assert_almost_equal(f_xy, f_xy_spemd, decimal=4)


if __name__ == '__main__':
    pytest.main()

