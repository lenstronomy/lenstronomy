__author__ = 'sibirrer'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['SIS']


class SIS(LensProfileBase):
    """
    this class contains the function and the derivatives of the Singular Isothermal Sphere


    .. math::
        \\kappa(x, y) = \\frac{1}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{x^2 + y^2}} \\right)

    with :math:`\\theta_{E}` is the Einstein radius,


    """
    param_names = ['theta_E', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, theta_E, center_x=0, center_y=0):
        x_shift = x - center_x
        y_shift = y - center_y
        f_ = theta_E * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        return f_

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            a = theta_E / max(0.000001, R)
        else:
            a=np.empty_like(R)
            r = R[R > 0]  #in the SIS regime
            a[R == 0] = 0
            a[R > 0] = theta_E / r
        f_x = a * x_shift
        f_y = a * y_shift
        return f_x, f_y

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        if isinstance(R, int) or isinstance(R, float):
            prefac = theta_E / max(0.000001, R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = theta_E / r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def rho2theta(rho0):
        """
        converts 3d density into 2d projected density parameter
        :param rho0:
        :param gamma:
        :return:
        """
        theta_E = np.pi * 2 * rho0
        return theta_E

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
    def mass_3d(r, rho0):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius in angular units
        :param rho0: density at angle=1
        :return: mass in angular units
        """
        mass_3d = 4 * np.pi * rho0 * r
        return mass_3d

    def mass_3d_lens(self, r, theta_E):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units

        :param r: radius in angular units
        :param theta_E: Einstein radius
        :return: mass in angular units
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_3d(r, rho0)

    def mass_2d(self, r, rho0):
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

    def mass_2d_lens(self, r, theta_E):
        """

        :param r: radius
        :param theta_E: Einstein radius
        :return: mass within a radius in projection
        """
        rho0 = self.theta2rho(theta_E)
        return self.mass_2d(r, rho0)

    def grav_pot(self, x, y, rho0, center_x=0, center_y=0):
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

    @staticmethod
    def density(r, rho0):
        """
        computes the density
        :param r: radius in angles
        :param rho0: density at angle=1
        :return: density at r
        """
        rho = rho0 / r**2
        return rho

    def density_lens(self, r, theta_E):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in projected in units of angles (i.e. arc seconds) results in the convergence quantity.

        :param r: 3d radius
        :param theta_E: Einstein radius
        :return: density(r)
        """
        rho0 = self.theta2rho(theta_E)
        return self.density(r, rho0)

    @staticmethod
    def density_2d(x, y, rho0, center_x=0, center_y=0):
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
