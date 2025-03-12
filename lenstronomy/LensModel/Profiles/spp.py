__author__ = "sibirrer"


import numpy as np
import scipy.special as special
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["SPP"]


class SPP(LensProfileBase):
    """Class to compute the Spherical Power-law Potential (SPP) Model.

    Given by:
    .. math::
        \\psi(r) = \\frac{2 E^2}{\\eta^2} \\left( \\frac{r^2 + s^2}{E^2} \\right)^{\\frac{\\eta}{2}}
    where:
    :math:`r^2 = (x-x_{\\text{center}})^2 + (y-y_{\\text{center}})^2` is squared radius from center of lens,
    :math:`s^2 = 0` due to no softening,
    :math:'E' is the characteristic scale factor related to the Einstein radius :math:`\\theta_{E}`, given by:
    .. math::
        E = \\frac{\\theta_E}{\\left( \\frac{3 - \\gamma}{2} \\right)^{\\frac{1}{1 - \\gamma}}}
    :math:'\\theta_{E}` is the Einstein radius of the lens,
    :math:`\\eta = -\\gamma + 3` is a parameter that depends on the power law slope :math:`\\gamma`,
    :math:`\\gamma` is the power law slope of the mass profile.
    """

    param_names = ["theta_E", "gamma", "center_x", "center_y"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "center_x": 100,
        "center_y": 100,
    }

    def function(self, x, y, theta_E, gamma, center_x=0, center_y=0):
        """
        :param x: set of x-coordinates
        :type x: array of size (n)
        :param y: set of y-coordinates
        :type y: array of size (n)
        :param theta_E: Einstein radius of lens
        :type theta_E: float
        :param gamma: power law slope of mass profile
        :type gamma: <2 float
        :param center_x: x-coordinate of the lens center
        :type center_x: float
        :param center_y: y-coordinate of the lens center
        :type center_y: float
        :returns: function value
        :raises: AttributeError, KeyError
        """
        gamma = self._gamma_limit(gamma)

        x_ = x - center_x
        y_ = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3

        p2 = x_**2 + y_**2
        s2 = 0.0  # softening
        return 2 * E**2 / eta**2 * ((p2 + s2) / E**2) ** (eta / 2)

    def derivatives(self, x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        """
        :param x: x-coordinate position
        :param y: y-coordinate position
        :param theta_E: Einstein radius of lens
        :param gamma: power law slope of mass profile
        :param center_x: x-coordinate of the lens center
        :param center_y: y-coordinate of the lens center
        :returns: f_x, f_y
        """
        gamma = self._gamma_limit(gamma)

        xt1 = x - center_x
        xt2 = y - center_y

        r2 = xt1 * xt1 + xt2 * xt2
        a = np.maximum(r2, 0.000001)
        r = np.sqrt(a)
        alpha = theta_E * (r2 / theta_E**2) ** (1 - gamma / 2.0)
        fac = alpha / r
        f_x = fac * xt1
        f_y = fac * xt2
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, center_x=0.0, center_y=0.0):
        """
        :param x: x-coordinate position
        :param y: y-coordinate position
        :param theta_E: Einstein radius of lens
        :param gamma: power law slope of mass profile
        :param center_x: x-coordinate of the lens center
        :param center_y: y-coordinate of the lens center
        :returns: f_xx, f_xy, f_yx, f_yy
        """
        gamma = self._gamma_limit(gamma)
        xt1 = x - center_x
        xt2 = y - center_y
        E = theta_E / ((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))
        # E = phi_E_spp
        eta = -gamma + 3.0

        P2 = xt1**2 + xt2**2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.000001, P2)
        else:
            a = np.empty_like(P2)
            p2 = P2[P2 > 0]  # in the SIS regime
            a[P2 == 0] = 0.000001
            a[P2 > 0] = p2

        kappa = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta - 2) * (xt1**2 + xt2**2) / a + (1 + 1))
        )
        gamma1 = (
            1.0
            / eta
            * (a / E**2) ** (eta / 2 - 1)
            * ((eta / 2 - 1) * (2 * xt1**2 - 2 * xt2**2) / a)
        )
        gamma2 = (
            4 * xt1 * xt2 * (1.0 / 2 - 1 / eta) * (a / E**2) ** (eta / 2 - 2) / E**2
        )

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def rho2theta(rho0, gamma):
        """Converts 3D density into 2D projected density parameter.

        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :returns: 2D projected density parameter (theta_E)
        """
        fac = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
            * rho0
        )

        # fac = theta_E**(gamma - 1)
        theta_E = fac ** (1.0 / (gamma - 1))
        return theta_E

    @staticmethod
    def theta2rho(theta_E, gamma):
        """Converts projected density parameter (in units of deflection) into 3d density
        parameter.

        :param theta_E: 2D projected density parameter
        :param gamma: power law slope of mass profile
        :returns: 3D density parameter (rho0)
        """
        fac1 = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * 2
            / (3 - gamma)
        )
        fac2 = theta_E ** (gamma - 1)
        rho0 = fac2 / fac1
        return rho0

    @staticmethod
    def mass_3d(r, rho0, gamma):
        """Calculates the mass enclosed in a 3D sphere of radius r.

        :param r: radius of the sphere
        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :returns: mass enclosed in the sphere
        """
        mass_3d = 4 * np.pi * rho0 / (-gamma + 3) * r ** (-gamma + 3)
        return mass_3d

    def mass_3d_lens(self, r, theta_E, gamma):
        """Calculates the mass enclosed in a 3D sphere of radius r using lens model
        parameters.

        :param r: radius of the sphere
        :param theta_E: 2D projected density parameter
        :param gamma: power law slope of mass profile
        :returns: mass enclosed in the sphere
        """
        rho0 = self.theta2rho(theta_E, gamma)
        return self.mass_3d(r, rho0, gamma)

    def mass_2d(self, r, rho0, gamma):
        """Calculates the mass enclosed in a projected circle of radius r.

        :param r: radius of the projected circle
        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :returns: mass enclosed in the projected circle
        """
        alpha = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (2 - gamma)
            / (3 - gamma)
            * 2
            * rho0
        )
        mass_2d = alpha * r * np.pi
        return mass_2d

    def mass_2d_lens(self, r, theta_E, gamma):
        """Calculates the mass enclosed in a projected circle of radius r using lens
        model parameters.

        :param r: radius of the projected circle
        :param theta_E: 2D projected density parameter
        :param gamma: power law slope of mass profile
        :returns: mass enclosed in the projected circle
        """
        rho0 = self.theta2rho(theta_E, gamma)
        return self.mass_2d(r, rho0, gamma)

    def grav_pot(self, x, y, rho0, gamma, center_x=0, center_y=0):
        """Gravitational potential (modulo 4 pi G and rho0 in appropriate units)

        :param x: x-coordinate position
        :param y: y-coordinate position
        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :param center_x: x-coordinate of the lens center
        :param center_y: y-coordinate of the lens center
        :returns: gravitational potential
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        mass_3d = self.mass_3d(r, rho0, gamma)
        pot = mass_3d / r
        return pot

    @staticmethod
    def density(r, rho0, gamma):
        """Calculates the 3D density.

        :param r: radius
        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :returns: 3D density
        """
        rho = rho0 / r**gamma
        return rho

    def density_lens(self, r, theta_E, gamma):
        """Calculates the 3D density using lens model parameters.

        The integral is projected in units of angles (i.e. arc seconds) results in the
        convergence quantity.

        :param r: radius
        :param theta_E: 2D projected density parameter
        :param gamma: power law slope of mass profile
        :returns: 3D density
        """
        rho0 = self.theta2rho(theta_E, gamma)
        return self.density(r, rho0, gamma)

    @staticmethod
    def density_2d(x, y, rho0, gamma, center_x=0, center_y=0):
        """Calculates the 2D projected density.

        :param x: x-coordinate position
        :param y: y-coordinate position
        :param rho0: 3D density parameter
        :param gamma: power law slope of mass profile
        :param center_x: x-coordinate of the center of the profile
        :param center_y: y-coordinate of the center of the profile
        :returns: 2D projected density
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        sigma = (
            np.sqrt(np.pi)
            * special.gamma(1.0 / 2 * (-1 + gamma))
            / special.gamma(gamma / 2.0)
            * r ** (1 - gamma)
            * rho0
        )
        return sigma

    @staticmethod
    def _gamma_limit(gamma):
        """Limits the power-law slope to certain bounds.

        :param gamma: power-law slope
        :return: bounded power-law slopte
        """
        return gamma
