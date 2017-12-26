from lenstronomy.LensModel.Profiles.composite_sersic_nfw import CompositeSersicNFW
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
import numpy.testing as npt


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





if __name__ == '__main__':
    pytest.main()