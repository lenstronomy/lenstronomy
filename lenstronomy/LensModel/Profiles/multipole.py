__author__ = "lynevdv"

import numpy as np

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Multipole", "EllipticalMultipole"]


class Multipole(LensProfileBase):
    """
    This class contains a SPHERICAL multipole contribution (for 1 component with m>=2)
    This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf, Equation B12
    Only the q=1 case (ie., spherical symmetry) makes this definition consistent with interpretation of multipoles as a deformation of the isophotes with an order m symmetry (eg., disky/boxy in the m=4 case).

    m : int, multipole order, m>=2
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """

    param_names = ["m", "a_m", "phi_m", "center_x", "center_y"]
    lower_limit_default = {
        "m": 2,
        "a_m": 0,
        "phi_m": -np.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def function(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Lensing potential of multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: lensing potential
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_ = r * a_m / (1 - m**2) * np.cos(m * (phi - phi_m))
        return f_

    def derivatives(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Deflection of a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
        Equation B12

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: deflection angles alpha_x, alpha_y
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_x = np.cos(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) + np.sin(
            phi
        ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))
        f_y = np.sin(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) - np.cos(
            phi
        ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))
        return f_x, f_y

    def hessian(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        """
        Hessian of a multipole contribution (for 1 component with m>=2)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: f_xx, f_xy, f_yx, f_yy
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        f_xx = 1.0 / r * np.sin(phi) ** 2 * a_m * np.cos(m * (phi - phi_m))
        f_yy = 1.0 / r * np.cos(phi) ** 2 * a_m * np.cos(m * (phi - phi_m))
        f_xy = -1.0 / r * a_m * np.cos(phi) * np.sin(phi) * np.cos(m * (phi - phi_m))
        return f_xx, f_xy, f_xy, f_yy


class EllipticalMultipole(LensProfileBase):
    """This class contains a multipole contribution that encode deviations from the
    elliptical isodensity contours of a SIE with any axis ratio q.

    m : int, multipole order, m=3 or m=4
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    q : axis ratio of the reference ellipses
    """

    param_names = ["m", "a_m", "phi_m", "q", "center_x", "center_y"]
    lower_limit_default = {
        "m": 2,
        "a_m": 0,
        "phi_m": -np.pi,
        "q": 0.001,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "q": 1,
        "center_x": 100,
        "center_y": 100,
    }

    def function(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0):
        """Lensing potential of multipole contribution (for 1 component with m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: lensing potential
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        if np.abs(1 - q) < 1e-4:  # avoid numerical instability when q is too close to 1
            f_ = r * a_m / (1 - m**2) * np.cos(m * (phi - phi_m))

        else:
            if m == 3:
                ### Symmetrize at this level to avoid discontinuities
                phisymm = param_util.cart2polar(-r * np.cos(phi), r * np.sin(phi))[1]
                phirot = (
                    param_util.cart2polar(-r * np.sin(phi), r * np.cos(phi))[1]
                    - np.pi / 2
                )  # find angle corresponding to phi in (-3pi/2, pi/2]
                phirotsymm = (
                    param_util.cart2polar(r * np.sin(phi), r * np.cos(phi))[1]
                    - np.pi / 2
                )  # find angle corresponding to -phi in (-3pi/2, pi/2]
                f_ = (
                    a_m
                    * np.sqrt(q)
                    * r
                    * (
                        (_F_m3_1(phi, q) - _F_m3_1(phisymm, q)) / 2 * np.cos(m * phi_m)
                        + (_F_m3_2(phirot, q) - _F_m3_2(phirotsymm, q))
                        / 2
                        * np.sin(m * phi_m)
                    )
                )

            elif m == 4:
                f_ = (
                    a_m
                    * np.sqrt(q)
                    * r
                    * (
                        _F_m4_1(phi, q=q) * np.cos(m * phi_m)
                        + _F_m4_2(phi, q=q) * np.sin(m * phi_m)
                    )
                )

            else:
                raise ValueError(
                    "Implementation of multipoles perturbation with m>4 for general axis ratio q not available."
                )

        return f_

    def derivatives(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0):
        """Deflection of a multipole contribution (for 1 component with m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: deflection angles alpha_x, alpha_y
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        if np.abs(1 - q) < 1e-4:  # avoid numerical instability when q is too close to 1
            f_x = np.cos(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) + np.sin(
                phi
            ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))
            f_y = np.sin(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) - np.cos(
                phi
            ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))

        else:
            if m == 3:
                ### Symmetrize at this level to avoid discontinuities
                phisymm = param_util.cart2polar(-r * np.cos(phi), r * np.sin(phi))[1]
                phirot = (
                    param_util.cart2polar(-r * np.sin(phi), r * np.cos(phi))[1]
                    - np.pi / 2
                )  # find angle corresponding to phi in (-3pi/2, pi/2]
                phirotsymm = (
                    param_util.cart2polar(r * np.sin(phi), r * np.cos(phi))[1]
                    - np.pi / 2
                )  # find angle corresponding to -phi in (-3pi/2, pi/2]
                alpha_x_1_pos, alpha_y_1_pos = _deflection_m3_base(
                    r, phi, q, a_3=a_m, phi_3=0
                )
                alpha_x_1_neg, alpha_y_1_neg = _deflection_m3_base(
                    r, phisymm, q, a_3=a_m, phi_3=0
                )
                alpha_x_2_pos, alpha_y_2_pos = _deflection_m3_base(
                    r, phirot, q, a_3=a_m, phi_3=np.pi / 6
                )
                alpha_x_2_neg, alpha_y_2_neg = _deflection_m3_base(
                    r, phirotsymm, q, a_3=a_m, phi_3=np.pi / 6
                )
                f_x = (
                    np.cos(m * phi_m) * (alpha_x_1_pos + alpha_x_1_neg) / 2
                    + np.sin(m * phi_m) * (alpha_x_2_pos - alpha_x_2_neg) / 2
                )
                f_y = (
                    np.cos(m * phi_m) * (alpha_y_1_pos - alpha_y_1_neg) / 2
                    + np.sin(m * phi_m) * (alpha_y_2_pos + alpha_y_2_neg) / 2
                )

            elif m == 4:
                F_m4 = _F_m4_1(phi, q=q) * np.cos(m * phi_m) + _F_m4_2(
                    phi, q=q
                ) * np.sin(m * phi_m)
                F_m4_prime = _F_m4_1_derivative(phi, q=q) * np.cos(
                    m * phi_m
                ) + _F_m4_2_derivative(phi, q=q) * np.sin(m * phi_m)
                f_x = a_m * np.sqrt(q) * (F_m4 * np.cos(phi) - F_m4_prime * np.sin(phi))
                f_y = a_m * np.sqrt(q) * (F_m4 * np.sin(phi) + F_m4_prime * np.cos(phi))

            else:
                raise ValueError(
                    "Implementation of multipoles perturbation with m>4 for general axis ratio q not available."
                )

        return f_x, f_y

    def hessian(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0):
        """Hessian of a multipole contribution (for 1 component with m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=2
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :return: f_xx, f_xy, f_yx, f_yy
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        phi_ell = np.angle(q * r * np.cos(phi) + 1j * r * np.sin(phi))
        R = np.sqrt(q * (r * np.cos(phi)) ** 2 + (r * np.sin(phi)) ** 2 / q)

        delta_r = a_m * np.cos(m * (phi_ell - phi_m)) * r / R
        f_xx = np.sin(phi) ** 2 * delta_r / r
        f_yy = np.cos(phi) ** 2 * delta_r / r
        f_xy = -np.sin(phi) * np.cos(phi) * delta_r / r
        return f_xx, f_xy, f_xy, f_yy


def _phi_ell(phi, q):
    return (
        phi
        - np.arctan2(np.sin(phi), np.cos(phi))
        + np.arctan2(np.sin(phi), q * np.cos(phi))
    )


def _F_m3_1(phi, q):
    term1 = np.cos(phi) * (
        q * (3 + q**2) * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        - (
            np.log(2) * (1 + q) ** 2
            - 2 * (1 - q) * (1 + q) ** 2 * (1 + np.log(2) / 4)
            + (1 - q**2) ** 2 / 4
        )
    )
    term2 = (
        2 * np.sin(phi) * (q * (q**2 + 3) * phi - (1 + 3 * q**2) * _phi_ell(phi, q))
    )  # Expression valid in (-pi, pi]
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _F_m3_1_derivative(phi, q):
    term1 = -np.cos(phi) * q * (3 + q**2) * 2 * (q**2 - 1) * np.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * np.cos(2 * phi)
    ) + np.sin(phi) * (
        -q * (3 + q**2) * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        + np.log(2) * (1 + q) ** 2
        - 2 * (1 - q) * (1 + q) ** 2 * (1 + np.log(2) / 4)
        + (1 - q**2) ** 2 / 4
    )
    term2 = 2 * np.cos(phi) * (
        q * (q**2 + 3) * phi - (1 + 3 * q**2) * _phi_ell(phi, q)
    ) + 2 * np.sin(phi) * (
        q * (q**2 + 3)
        - (1 + 3 * q**2) * q / (q**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2)
    )  # Expression valid in (-pi, pi]
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _F_m3_2(phi, q):
    # Expression valid in (-3*pi/2, pi/2]
    return 1 / q * _F_m3_1(phi + np.pi / 2, 1 / q)


def _F_m3_2_derivative(phi, q):
    # Expression valid in (-3*pi/2, pi/2]
    return 1 / q * _F_m3_1_derivative(phi + np.pi / 2, 1 / q)


def _deflection_m3_base(r, phi, q, a_3, phi_3):
    F_m3 = _F_m3_1(phi, q=q) * np.cos(3 * phi_3) + _F_m3_2(phi, q=q) * np.sin(3 * phi_3)
    F_m3_prime = _F_m3_1_derivative(phi, q=q) * np.cos(3 * phi_3) + _F_m3_2_derivative(
        phi, q=q
    ) * np.sin(3 * phi_3)
    alpha_x = a_3 * np.sqrt(q) * (F_m3 * np.cos(phi) - F_m3_prime * np.sin(phi))
    alpha_y = a_3 * np.sqrt(q) * (F_m3 * np.sin(phi) + F_m3_prime * np.cos(phi))
    return alpha_x, alpha_y


def _F_m4_1(phi, q):
    term1 = (
        -4
        * np.sqrt(2)
        * (1 + 4 * q**2 + q**4 + (q**4 - 1) * np.cos(2 * phi))
        / ((3 * (1 - q**2) ** 2) * np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi)))
    )
    term2 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * np.cos(phi)
        * np.arctan(
            (np.sqrt(2 * (1 - q**2)) * np.cos(phi))
            / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        )
    )
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * np.sin(phi)
        * np.log(
            np.sqrt(1 - q**2) * np.sin(phi) / q
            + np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


def _F_m4_1_derivative(phi, q):
    term1 = (
        -4
        * np.sqrt(2)
        * (1 + q**4 + (q**4 - 1) * np.cos(2 * phi))
        * np.sin(2 * phi)
        / (3 * (1 - q**2) * (1 + q**2 + (q**2 - 1) * np.cos(2 * phi)) ** (3 / 2))
    )
    term2 = (
        -(1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * (
            np.sin(phi)
            * np.arctan(
                (np.sqrt(2 * (1 - q**2)) * np.cos(phi))
                / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
            )
            + np.sqrt(2 * (1 - q**2))
            * np.sin(2 * phi)
            / (2 * np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi)))
        )
    )
    term3 = (
        (1 + 6 * q**2 + q**4)
        / (1 - q**2) ** (5 / 2)
        * np.cos(phi)
        * (
            np.log(
                np.sqrt(1 - q**2) * np.sin(phi) / q
                + np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
            )
            + np.sqrt(1 - q**2)
            / q
            * np.sin(phi)
            / np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


def _F_m4_2(phi, q):
    term1 = (
        -4
        * np.sqrt(2)
        * q
        / (3 * (1 - q**2))
        * np.sin(2 * phi)
        / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
    )
    term2 = (
        -4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * np.sin(phi)
        * np.arctan(
            (np.sqrt(2 * (1 - q**2)) * np.cos(phi))
            / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        )
    )
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * np.cos(phi)
        * np.log(
            np.sqrt(1 - q**2) * np.sin(phi) / q
            + np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3


def _F_m4_2_derivative(phi, q):
    term1 = (
        -8
        * np.sqrt(2)
        * q
        / (6 * (1 - q**2))
        * (
            -(1 - q**2)
            * np.sin(2 * phi) ** 2
            / (1 + q**2 + (q**2 - 1) * np.cos(2 * phi)) ** (3 / 2)
            + 2 * np.cos(2 * phi) / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        )
    )
    term2 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -np.cos(phi)
            * np.arctan(
                (np.sqrt(2 * (1 - q**2)) * np.cos(phi))
                / np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
            )
            + 2
            * np.sqrt(2 * (1 - q**2))
            * np.sin(phi) ** 2
            / (2 * np.sqrt(1 + q**2 + (q**2 - 1) * np.cos(2 * phi)))
        )
    )
    term3 = (
        4
        * q
        * (1 + q**2)
        / (1 - q**2) ** (5 / 2)
        * (
            -np.sin(phi)
            * np.log(
                np.sqrt(1 - q**2) * np.sin(phi) / q
                + np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
            )
            + np.sqrt(1 - q**2)
            / q
            * np.cos(phi) ** 2
            / np.sqrt(1 + (1 - q**2) / q**2 * np.sin(phi) ** 2)
        )
    )

    return term1 + term2 + term3
