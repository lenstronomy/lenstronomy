from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ['SIE']


class SIE(LensProfileBase):
    """
    class for singular isothermal ellipsoid (SIS with ellipticity)

    .. math::
        \\kappa(x, y) = \\frac{1}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate sys- tem aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{1}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 − e*\\cos(2*\\phi)}} \\right)

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.

    """
    param_names = ['theta_E', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, NIE=True):
        """

        :param NIE: bool, if True, is using the NIE analytic model. Otherwise it uses PEMD with gamma=2 from fastell4py
        """
        self._nie = NIE
        if NIE:
            from lenstronomy.LensModel.Profiles.nie import NIE
            self.profile = NIE()
        else:
            from lenstronomy.LensModel.Profiles.epl import EPL
            self.profile = EPL()
        self._s_scale = 0.0000000001
        self._gamma = 2
        super(SIE, self).__init__()

    def function(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.function(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.function(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)

    def derivatives(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.derivatives(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.derivatives(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)

    def hessian(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.hessian(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.hessian(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)

    @staticmethod
    def theta2rho(theta_E):
        """
        converts projected density parameter (in units of deflection) into 3d density parameter
        :param theta_E:
        :return:
        """
        fac1 = np.pi * 2
        rho0 = theta_E / fac1
        return rho0

    @staticmethod
    def mass_3d(r, rho0, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius in angular units
        :param rho0: density at angle=1
        :return: mass in angular units
        """
        mass_3d = 4 * np.pi * rho0 * r
        return mass_3d

    def mass_3d_lens(self, r, theta_E, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units

        :param r: radius in angular units
        :param theta_E: Einstein radius
        :return: mass in angular units
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_3d(r, rho0)

    def mass_2d(self, r, rho0, e1=0, e2=0):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        alpha = np.pi * np.pi * 2 * rho0
        mass_2d = alpha*r
        return mass_2d

    def mass_2d_lens(self, r, theta_E, e1=0, e2=0):
        """

        :param r:
        :param theta_E:
        :return:
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_2d(r, rho0)

    def grav_pot(self, x, y, rho0, e1=0, e2=0, center_x=0, center_y=0):
        """
        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        mass_3d = self.mass_3d(r, rho0)
        pot = mass_3d/r
        return pot

    def density_lens(self, r, theta_E, e1=0, e2=0):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius in angles
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: density
        """
        rho0 = self.theta2rho(theta_E)
        return self.density(r, rho0)

    @staticmethod
    def density(r, rho0, e1=0, e2=0):
        """
        computes the density
        :param r: radius in angles
        :param rho0: density at angle=1
        :return: density at r
        """
        rho = rho0 / r**2
        return rho

    @staticmethod
    def density_2d(x, y, rho0, e1=0, e2=0, center_x=0, center_y=0):
        """
        projected density
        :param x:
        :param y:
        :param rho0:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma = np.pi * rho0 / r
        return sigma
