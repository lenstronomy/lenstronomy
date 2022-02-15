__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.p_jaffe_ellipse import PJaffe_Ellipse
import lenstronomy.Util.param_util as param_util

import numpy as np
import numpy.testing as npt
import pytest

class TestP_JAFFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.profile = PJaffe_Ellipse()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.profile.function(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.9060417038170876, decimal=8)
        x = np.array([0])
        y = np.array([0])

        values = self.profile.function(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.20267440905756931, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function( x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0.83655317460063972, decimal=8)
        npt.assert_almost_equal(values[1], 1.0291644086604779, decimal=8)
        npt.assert_almost_equal(values[2], 1.1938735098090378, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(f_x[0], 0.084067857192459253, decimal=8)
        npt.assert_almost_equal(f_y[0], 0.25220357157737772, decimal=8)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.profile.derivatives(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0][0], 0.084067857192459253, decimal=8)
        npt.assert_almost_equal(values[1][0], 0.25220357157737772, decimal=8)
        npt.assert_almost_equal(values[0][1], 0.17861253965407037, decimal=8)
        npt.assert_almost_equal(values[1][1], 0.08930626982703517, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        q, phi_G = 0.8, 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(f_xx[0], 0.064049581333103234, decimal=8)
        npt.assert_almost_equal(f_yy[0], -0.054062473386906618, decimal=8)
        npt.assert_almost_equal(f_xy[0], -0.060054723013958089, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=6)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.profile.hessian(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0][0], 0.064049581333103234, decimal=8)
        npt.assert_almost_equal(values[3][0], -0.054062473386906618, decimal=8)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, sigma0=1, Ra=0.5, Rs=0.8, e1=0, e2=0)
        npt.assert_almost_equal(mass, 0.87077306005349242, decimal=8)


if __name__ == '__main__':
    pytest.main()
