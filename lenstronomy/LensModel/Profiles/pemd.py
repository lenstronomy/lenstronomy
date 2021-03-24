__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.spemd import SPEMD
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['PEMD']


class PEMD(LensProfileBase):
    """
    class for power law ellipse mass density profile.
    This class effectively calls the class SPEMD_SMOOTH with a fixed and very small central smoothing scale
    to perform the numerical integral using the FASTELL code by Renan Barkana.

    .. math::
        \\kappa(x, y) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)^{\\gamma-1}

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`\\gamma` is the negative power-law slope of the 3D mass distributions,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate system aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 − e*\\cos(2*\\phi)}} \\right)^{\\gamma-1}

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.


    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, suppress_fastell=False):
        """

        :param suppress_fastell: bool, if True, does not raise if fastell4py is not installed
        """
        self._s_scale = 0.0001  # smoothing scale as used to numerically compute a power-law profile
        self.spp = SPP()
        self.spemd_smooth = SPEMD(suppress_fastell=suppress_fastell)
        super(PEMD, self).__init__()

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: lensing potential
        """
        return self.spemd_smooth.function(x, y, theta_E, gamma, e1, e2, self._s_scale, center_x, center_y)

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: deflection angles alpha_x, alpha_y
        """
        return self.spemd_smooth.derivatives(x, y, theta_E, gamma, e1, e2, self._s_scale, center_x, center_y)

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param theta_E: Einstein radius (angle), pay attention to specific definition!
        :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-position of lens center
        :param center_y: y-position of lens center
        :return: Hessian components f_xx, f_xy, f_yx, f_yy
        """
        return self.spemd_smooth.hessian(x, y, theta_E, gamma, e1, e2, self._s_scale, center_x, center_y)

    def mass_3d_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """
        computes the spherical power-law mass enclosed (with SPP routine)
        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.density_lens(r, theta_E, gamma)
