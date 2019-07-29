__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE

import numpy as np
import numpy.testing as npt
import pytest


class TestNFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        alpha_Rs = self.nfw._rho02alpha(rho0, Rs)
        values = self.nfw.function(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values[0], 2.4764530888727556, decimal=5)
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        rho0 = 1
        alpha_Rs = self.nfw._rho02alpha(rho0, Rs)
        values = self.nfw.function(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values[0], 0, decimal=4)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.nfw.function(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values[0], 2.4764530888727556, decimal=5)
        npt.assert_almost_equal(values[1], 3.5400250357511416, decimal=5)
        npt.assert_almost_equal(values[2], 4.5623722261790647, decimal=5)

    def test_derivatives(self):
        Rs = .1
        alpha_Rs = 0.0122741127776
        x_array = np.array([0.0, 0.00505050505,0.0101010101,0.0151515152,0.0202020202,0.0252525253,
            0.0303030303,0.0353535354,0.0404040404,0.0454545455,0.0505050505,0.0555555556,0.0606060606,0.0656565657,0.0707070707,0.0757575758,0.0808080808,0.0858585859,0.0909090909,0.095959596,0.101010101,0.106060606,
            0.111111111,0.116161616,0.121212121,0.126262626,0.131313131,0.136363636,0.141414141,0.146464646,0.151515152,0.156565657,
            0.161616162,0.166666667,0.171717172,0.176767677,0.181818182,0.186868687,0.191919192,0.196969697,0.202020202,0.207070707,0.212121212,0.217171717,0.222222222,0.227272727,0.232323232,0.237373737,0.242424242,0.247474747,0.252525253,0.257575758,0.262626263,0.267676768,0.272727273,0.277777778,0.282828283,
            0.287878788,0.292929293,0.297979798,0.303030303,0.308080808,0.313131313,0.318181818,0.323232323,0.328282828,0.333333333,0.338383838,0.343434343,0.348484848,
            0.353535354,0.358585859,0.363636364,0.368686869,0.373737374,0.378787879,0.383838384,0.388888889,0.393939394,0.398989899,0.404040404,0.409090909,
            0.414141414,0.419191919,0.424242424,0.429292929,0.434343434,0.439393939,0.444444444,0.449494949,0.454545455,0.45959596,0.464646465,0.46969697,0.474747475,0.47979798,0.484848485,0.48989899,0.494949495,0.5])
        truth_alpha = np.array([0.0, 0.00321693283, 0.00505903212,
            0.00640987376,0.00746125453,0.00830491158, 0.00899473755, 0.00956596353,0.0100431963,0.0104444157,0.0107831983,0.0110700554,0.0113132882,0.0115195584,0.0116942837,0.0118419208,
            0.011966171,0.0120701346,0.012156428,0.0122272735,0.0122845699,0.0123299487,0.0123648177,0.0123903978,0.0124077515,0.0124178072,0.0124213787,0.0124191816,0.0124118471,0.0123999334,0.0123839353,0.0123642924,0.0123413964,
            0.0123155966,0.0122872054,0.0122565027,0.0122237393,0.0121891409,0.0121529102,0.0121152302,0.0120762657,0.0120361656,0.0119950646,0.0119530846,0.0119103359,0.0118669186,0.0118229235,0.0117784329,0.0117335217,
            0.011688258,0.0116427037,0.0115969149,0.0115509429,0.0115048343,0.0114586314,0.0114123729,0.011366094,0.0113198264,0.0112735995,0.0112274395,0.0111813706,0.0111354147,
            0.0110895915,0.011043919,0.0109984136,0.01095309,0.0109079617,0.0108630406,0.0108183376,0.0107738625,0.010729624,0.01068563,0.0106418875,0.0105984026,0.0105551809,0.0105122271,0.0104695455,0.0104271398,0.010385013,0.0103431679,0.0103016067,0.0102603311,
            0.0102193428,0.0101786427,0.0101382318,0.0100981105,0.0100582792,0.0100187377,0.00997948602,0.00994052364,0.00990184999,
            0.00986346433, 0.00982536573,0.00978755314, 0.00975002537, 0.0097127811, 0.00967581893, 0.00963913734, 0.00960273473, 0.00956660941])
        y_array = np.zeros_like(x_array)
        f_x, f_y = self.nfw.derivatives(x_array, y_array, Rs, alpha_Rs)
        #print(f_x/truth_alpha)
        for i in range(len(x_array)):
            npt.assert_almost_equal(f_x[i], truth_alpha[i], decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        alpha_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_xx, f_yy,f_xy = self.nfw.hessian(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(f_xx[0], 0.40855527280658294, decimal=5)
        npt.assert_almost_equal(f_yy[0], 0.037870368296371637, decimal=5)
        npt.assert_almost_equal(f_xy[0], -0.2471232696734742, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.nfw.hessian(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values[0][0], 0.40855527280658294, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.037870368296371637, decimal=5)
        npt.assert_almost_equal(values[2][0], -0.2471232696734742, decimal=5)
        npt.assert_almost_equal(values[0][1], -0.046377502475445781, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.30577812878681554, decimal=5)
        npt.assert_almost_equal(values[2][1], -0.13205836172334798, decimal=5)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        m_3d = self.nfw.mass_3d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_3d, 1.1573795105019022, decimal=8)

    def test_interpol(self):
        Rs = 3
        alpha_Rs = 1
        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])

        nfw = NFW(interpol=False)
        nfw_interp = NFW(interpol=True)
        nfw_interp_lookup = NFW(interpol=True, lookup=True)

        values = nfw.function(x, y, Rs, alpha_Rs)
        values_interp = nfw_interp.function(x, y, Rs, alpha_Rs)
        values_interp_lookup = nfw_interp_lookup.function(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values, values_interp, decimal=4)
        npt.assert_almost_equal(values, values_interp_lookup, decimal=4)

        values = nfw.derivatives(x, y, Rs, alpha_Rs)
        values_interp = nfw_interp.derivatives(x, y, Rs, alpha_Rs)
        values_interp_lookup = nfw_interp_lookup.derivatives(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values, values_interp, decimal=4)
        npt.assert_almost_equal(values, values_interp_lookup, decimal=4)

        values = nfw.hessian(x, y, Rs, alpha_Rs)
        values_interp = nfw_interp.hessian(x, y, Rs, alpha_Rs)
        values_interp_lookup = nfw_interp_lookup.hessian(x, y, Rs, alpha_Rs)
        npt.assert_almost_equal(values, values_interp, decimal=4)
        npt.assert_almost_equal(values, values_interp_lookup, decimal=4)



class TestMassAngleConversion(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        self.nfw = NFW()
        self.nfw_ellipse = NFW_ELLIPSE()

    def test_angle(self):
        x, y = 1, 0
        alpha1, alpha2 = self.nfw.derivatives(x, y, alpha_Rs=1., Rs=1.)
        assert alpha1 == 1.

    def test_convertAngle2rho(self):
        rho0 = self.nfw._alpha2rho0(alpha_Rs=1., Rs=1.)
        assert rho0 == 0.81472283831773229

    def test_convertrho02angle(self):
        alpha_Rs_in = 1.5
        Rs = 1.5
        rho0 = self.nfw._alpha2rho0(alpha_Rs=alpha_Rs_in, Rs=Rs)
        alpha_Rs_out = self.nfw._rho02alpha(rho0, Rs)
        assert alpha_Rs_in == alpha_Rs_out


if __name__ == '__main__':
    pytest.main()
