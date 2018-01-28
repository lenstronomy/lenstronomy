from lenstronomy.LensModel.Profiles.composite_sersic_nfw import CompositeSersicNFW
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
import numpy.testing as npt
import pytest


class TestMassAngleConversion(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        self.composite = CompositeSersicNFW()
        self.sersic = Sersic()
        self.nfw = NFW_ELLIPSE()

    def test_convert(self):
        theta_E = 1.
        mass_light = 1/2.
        Rs = 5.
        n_sersic = 2.
        r_eff = 0.7
        theta_Rs, k_eff = self.composite.convert_mass(theta_E, mass_light, Rs, n_sersic, r_eff)

        alpha_E_sersic, _ = self.sersic.derivatives(theta_E, 0, n_sersic, r_eff, k_eff=1)
        alpha_E_nfw, _ = self.nfw.derivatives(theta_E, 0, Rs, theta_Rs=1, q=1, phi_G=0)
        a = theta_Rs * alpha_E_nfw + (k_eff * alpha_E_sersic)
        b = theta_Rs * alpha_E_nfw / (k_eff * alpha_E_sersic)
        npt.assert_almost_equal(a, theta_E, decimal=10)
        npt.assert_almost_equal(b, mass_light, decimal=10)

    def test_function(self):
        theta_E = 1.
        mass_light = 1/2.
        Rs = 5.
        n_sersic = 2.
        r_eff = 0.7
        q, phi_G = 0.9, 0
        q_s, phi_G_s = 0.7, 0.5
        x, y, = 1, 1
        f_ = self.composite.function(x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, q_s, phi_G_s, center_x=0, center_y=0)
        npt.assert_almost_equal(f_, 1.1983595285200526, decimal=10)

    def test_derivatives(self):
        theta_E = 1.
        mass_light = 1/2.
        Rs = 5.
        n_sersic = 2.
        r_eff = 0.7
        q, phi_G = 0.9, 0
        q_s, phi_G_s = 0.7, 0.5
        x, y, = 1, 1
        f_x, f_y = self.composite.derivatives(x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, q_s, phi_G_s, center_x=0, center_y=0)
        npt.assert_almost_equal(f_x, 0.54138666294863724, decimal=10)
        npt.assert_almost_equal(f_y, 0.75841883763728535, decimal=10)

    def test_hessian(self):
        theta_E = 1.
        mass_light = 1/2.
        Rs = 5.
        n_sersic = 2.
        r_eff = 0.7
        q, phi_G = 0.9, 0
        q_s, phi_G_s = 0.7, 0.5
        x, y, = 1, 1
        f_xx, f_yy, f_xy = self.composite.hessian(x, y, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, q_s, phi_G_s, center_x=0, center_y=0)
        npt.assert_almost_equal(f_xx, 0.43275276043197586, decimal=10)
        npt.assert_almost_equal(f_yy, 0.37688935994317774, decimal=10)
        npt.assert_almost_equal(f_xy, -0.46895575042671389, decimal=10)


if __name__ == '__main__':
    pytest.main()