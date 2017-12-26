__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH


class SPEMD(object):
    """
    class for smooth power law ellipse mass density profile
    """
    def __init__(self):
        self.s2 = 0.00000001
        self.spp = SPP()
        self.spemd_smooth = SPEMD_SMOOTH()

    def function(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.spemd_smooth.function(x, y, theta_E, gamma, q, phi_G, self.s2, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.spemd_smooth.derivatives(x, y, theta_E, gamma, q, phi_G, self.s2, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, q, phi_G, center_x=0, center_y=0):
        return self.spemd_smooth.hessian(x, y, theta_E, gamma, q, phi_G, self.s2, center_x, center_y)

    def mass_3d_lens(self, r, theta_E, gamma, q, phi_G):
        """
        computes the spherical power-law mass enclosed (with SPP routiune)
        :param r:
        :param theta_E:
        :param gamma:
        :param q:
        :param phi_G:
        :return:
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def convert_params(self, theta_E, gamma, q):
        """

        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :return:   prefactor to SPEMP profile for FASTELL
        """
        return self.spemd_smooth.convert_params(theta_E, gamma, q)