from lenstronomy.LensModel.Profiles.flexion import Flexion
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Flexionfg"]


class Flexionfg(LensProfileBase):
    """
    Flexion consist of basis F flexion and G flexion (F1,F2,G1,G2),
    see formulas 2.54, 2.55 in Massimo Meneghetti 2017 - "Introduction to Gravitational
    Lensing".

    The flexion is a third order derivative of the lensing potential, i.e. the deflection angle.
    The flexion is a measure of the lensing shear gradient, i.e. the change of the shear field
    with respect to the position on the sky.

    The second order lensing effects can be expressed in terms of the third derivatives of the lensing potential,
    which can be written in terms of the flexion F and G.

    .. math::
      \\beta_i\\simeq\\sum_{j} A_{ij}\\theta_j+\\frac{1}{2}\\sum_{j}\\sum_{k}D_{ijk}\\theta_j\\theta_k

    Which is the second order lensing effects expressed in terms of the third derivatives of the lensing potential
    (Formula 3.60 in Meneghetti 2021).

    These in turn can be expressed in terms of the flexion F and G.

    .. math::
        D_{111}=-2\\gamma_{11}-\\gamma_{22}=\\frac{1}{2}(3F_1+G_1)
        D_{211}=D_{131}=D_{112}-\\gamma_{21}=-\\frac{1}{2}(F_2+G_2)
        D_{122}=D_{212}=D_{221}=-\\gamma_{22}=-\\frac{1}{2}(F_1-G_1)
        D_{222}=2\\gamma_{12}-\\gamma_{21}=-\\frac{1}{2}(3F_2-G_2)

    (Formula 3.98 in Meneghetti 2017).

    Then we find that the two components of \\vec{\\beta} are:

    .. math::
        \\beta_1=A_{11}\\theta_1+A_{12}\\theta_2+\\frac{1}{2}D_{111}\\theta_1^2+D_{121}\\theta_1\\theta_2+\\frac{1}{2}D_{122}\\theta_2^2
        \\beta_2=A_{21}\\theta_1+A_{22}\\theta_2+\\frac{1}{2}D_{211}\\theta_1^2+D_{212}\\theta_1\\theta_2+\\frac{1}{2}D_{222}\\theta_2^2

    (Formula 3.99 in Meneghetti 2021).

    Now we can express the flexion in terms of ra_0, and dec_0,
    which are the zero-points of the polynomial expansion.
    Instead of using absolute coordinates \\theta_1, and \\theta_2,
    we define the relative angular positions:

    .. math::
        x = \\theta_1 - ra_0
        y = \\theta_2 - dec_0

    """

    param_names = ["F1", "F2", "G1", "G2", "ra_0", "dec_0"]
    lower_limit_default = {
        "F1": -0.1,
        "F2": -0.1,
        "G1": -0.1,
        "G2": -0.1,
        "ra_0": -100,
        "dec_0": -100,
    }
    upper_limit_default = {
        "F1": 0.1,
        "F2": 0.1,
        "G1": 0.1,
        "G2": 0.1,
        "ra_0": 100,
        "dec_0": 100,
    }

    def __init__(self):
        self.flexion_cart = Flexion()
        super(Flexionfg, self).__init__()

    def function(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """Lensing potential.

        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center y-coordinate
        :return: lensing potential
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.function(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def derivatives(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """Deflection angle.

        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center x-coordinate
        :return: deflection angle.
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.derivatives(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def hessian(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """Hessian matrix.

        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center y-coordinate
        :return: second order derivatives f_xx, f_yy, f_xy
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.hessian(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    @staticmethod
    def transform_fg(F1, F2, G1, G2):
        """Basis transform from (F1,F2,G1,G2) to (g1,g2,g3,g4).

        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :return: g1,g2,g3,g4 (phi_xxx, phi_xxy, phi_xyy, phi_yyy)
        """
        g1 = (3 * F1 + G1) * 0.5
        g2 = (3 * F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        return g1, g2, g3, g4
