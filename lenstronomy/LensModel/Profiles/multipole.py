__author__ = "lynevdv"

import numpy as np

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Multipole", "EllipticalMultipole"]


class Multipole(LensProfileBase):
    """
    This class contains a CIRCULAR multipole contribution (for 1 component with m>=2)
    This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf, Equation B12
    Only the q=1 case (ie., circular symmetry) makes this definition consistent with interpretation of multipoles as a deformation of the isophotes with an order m symmetry (eg., disky/boxy in the m=4 case).

    m : int, multipole order, m>=1
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """

    param_names = ["m", "a_m", "phi_m", "center_x", "center_y", "r_E"]
    lower_limit_default = {
        "m": 1,
        "a_m": 0,
        "phi_m": -np.pi,
        "center_x": -100,
        "center_y": -100,
        "r_E": 0,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "center_x": 100,
        "center_y": 100,
        "r_E": 100,
    }

    def function(self, x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Lensing potential of multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for the m=1, Einstein radius by default)
        :return: lensing potential
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        if m == 1:
            r = np.maximum(r, 0.000001)
            f_ = r * np.log(r / r_E) * a_m / 2 * np.cos(phi - phi_m)
        else:
            f_ = r * a_m / (1 - m**2) * np.cos(m * (phi - phi_m))
        return f_

    def derivatives(self, x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Deflection of a multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf
        Equation B12

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for the m=1, Einstein radius by default)
        :return: deflection angles alpha_x, alpha_y
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)

        if m == 1:
            r = np.maximum(r, 0.000001)
            f_x = (
                a_m
                / 2
                * (np.cos(phi_m) * np.log(r / r_E) + np.cos(phi - phi_m) * np.cos(phi))
            )
            f_y = (
                a_m
                / 2
                * (np.sin(phi_m) * np.log(r / r_E) + np.cos(phi - phi_m) * np.sin(phi))
            )
        else:
            f_x = np.cos(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) + np.sin(
                phi
            ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))
            f_y = np.sin(phi) * a_m / (1 - m**2) * np.cos(m * (phi - phi_m)) - np.cos(
                phi
            ) * m * a_m / (1 - m**2) * np.sin(m * (phi - phi_m))
        return f_x, f_y

    def hessian(self, x, y, m, a_m, phi_m, center_x=0, center_y=0, r_E=1):
        """
        Hessian of a multipole contribution (for 1 component with m>=1)
        This uses the same definitions as Xu et al.(2013) in Appendix B3 https://arxiv.org/pdf/1307.4220.pdf

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order, m>=1
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (not used for Hessian)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        if m == 1:
            f_xx = (
                a_m
                / (2 * r)
                * (
                    2 * np.cos(phi_m) * np.cos(phi)
                    - np.cos(phi - phi_m) * np.cos(2 * phi)
                )
            )
            f_yy = (
                a_m
                / (2 * r)
                * (
                    2 * np.sin(phi_m) * np.sin(phi)
                    + np.cos(phi - phi_m) * np.cos(2 * phi)
                )
            )
            f_xy = (
                a_m
                / (2 * r)
                * (np.sin(phi + phi_m) - np.cos(phi - phi_m) * np.sin(2 * phi))
            )
        else:
            f_xx = 1.0 / r * np.sin(phi) ** 2 * a_m * np.cos(m * (phi - phi_m))
            f_yy = 1.0 / r * np.cos(phi) ** 2 * a_m * np.cos(m * (phi - phi_m))
            f_xy = (
                -1.0 / r * a_m * np.cos(phi) * np.sin(phi) * np.cos(m * (phi - phi_m))
            )
        return f_xx, f_xy, f_xy, f_yy


class EllipticalMultipole(LensProfileBase):
    """This class contains a multipole contribution that encode deviations from the
    elliptical isodensity contours of a SIE with any axis ratio q.

    m : int, multipole order, (m=1, m=3 or m=4)
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    q : axis ratio of the reference ellipses
    """

    param_names = ["m", "a_m", "phi_m", "q", "center_x", "center_y", "r_E"]
    lower_limit_default = {
        "m": 1,
        "a_m": 0,
        "phi_m": -np.pi,
        "q": 0.001,
        "center_x": -100,
        "center_y": -100,
        "r_E": 0,
    }
    upper_limit_default = {
        "m": 100,
        "a_m": 100,
        "phi_m": np.pi,
        "q": 1,
        "center_x": 100,
        "center_y": 100,
        "r_E": 100,
    }

    def function(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Lensing potential of multipole contribution (for 1 component with m=1, m=3 or
        m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for odd m, Einstein radius by
            default)
        :return: lensing potential
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)

        if (
            np.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8
        ):  # avoid numerical instability when q is too close to 1 by taking circular multipole solution
            sph_multipole = Multipole()
            f_ = sph_multipole.function(
                x, y, m, a_m, phi_m, center_x=center_x, center_y=center_y, r_E=r_E
            )

        else:
            if m == 1:
                f_ = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * _potential_m1_1(r, phi, q, r_E)
                        - (1 / q)
                        * np.sin(m * phi_m)
                        * _potential_m1_1(r, phi + np.pi / 2, 1 / q, r_E)
                    )
                )

            elif m == 3:
                f_ = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * _potential_m3_1(r, phi, q, r_E)
                        + (1 / q)
                        * np.sin(m * phi_m)
                        * _potential_m3_1(r, phi + np.pi / 2, 1 / q, r_E)
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
                    "Implementation of multipoles perturbation for general axis ratio q only available for m=1, m=3 or m=4."
                )

        return f_

    def derivatives(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Deflection of a multipole contribution (for 1 component with m=1, m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (only used for odd m, Einstein radius by
            default)
        :return: deflection angles alpha_x, alpha_y
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)

        if (
            np.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8
        ):  # avoid numerical instability when q is too close to 1 by taking circular multipole solution
            sph_multipole = Multipole()
            f_x, f_y = sph_multipole.derivatives(
                x, y, m, a_m, phi_m, center_x=center_x, center_y=center_y, r_E=r_E
            )

        else:
            if m == 1:
                alpha_x_1, alpha_y_1 = _alpha_m1_1(r, phi, q, r_E)
                alpha_x_2, alpha_y_2 = _alpha_m1_1(r, phi + np.pi / 2, 1 / q, r_E)
                f_x = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * alpha_x_1
                        - (1 / q) * np.sin(m * phi_m) * alpha_y_2
                    )
                )
                f_y = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * alpha_y_1
                        + (1 / q) * np.sin(m * phi_m) * alpha_x_2
                    )
                )

            elif m == 3:
                alpha_x_1, alpha_y_1 = _alpha_m3_1(r, phi, q, r_E)
                alpha_x_2, alpha_y_2 = _alpha_m3_1(r, phi + np.pi / 2, 1 / q, r_E)
                f_x = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * alpha_x_1
                        + (1 / q) * np.sin(m * phi_m) * alpha_y_2
                    )
                )
                f_y = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * alpha_y_1
                        - (1 / q) * np.sin(m * phi_m) * alpha_x_2
                    )
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
                    "Implementation of multipoles perturbation for general axis ratio q only available for m=1, m=3 or m=4."
                )

        return f_x, f_y

    def hessian(self, x, y, m, a_m, phi_m, q, center_x=0, center_y=0, r_E=1):
        """Hessian of a multipole contribution (for 1 component with m=1, m=3 or m=4)

        :param x: x-coordinate to evaluate function
        :param y: y-coordinate to evaluate function
        :param m: int, multipole order (m=1, m=3 or m=4)
        :param a_m: float, multipole strength
        :param phi_m: float, multipole orientation in radian
        :param center_x: x-position
        :param center_y: y-position
        :param r_E: float, normalizing radius (not used for Hessian)
        :return: f_xx, f_xy, f_yx, f_yy
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)

        if (
            np.abs(1 - q**2) ** ((m + 1) / 2) < 1e-8
        ):  # avoid numerical instability when q is too close to 1 by taking circular multipole solution
            sph_multipole = Multipole()
            f_xx, f_xy, f_xy, f_yy = sph_multipole.hessian(
                x, y, m, a_m, phi_m, center_x=center_x, center_y=center_y, r_E=r_E
            )

        else:
            if m == 1:
                d2psi_dx2_1, d2psi_dy2_1, d2psi_dxdy_1 = _hessian_m1_1(r, phi, q)
                d2psi_dx2_2, d2psi_dy2_2, d2psi_dxdy_2 = _hessian_m1_1(
                    r, phi + np.pi / 2, 1 / q
                )
                f_xx = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dx2_1
                        - (1 / q) * np.sin(m * phi_m) * d2psi_dy2_2
                    )
                )
                f_yy = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dy2_1
                        - (1 / q) * np.sin(m * phi_m) * d2psi_dx2_2
                    )
                )
                f_xy = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dxdy_1
                        + (1 / q) * np.sin(m * phi_m) * d2psi_dxdy_2
                    )
                )

            elif m == 3:
                d2psi_dx2_1, d2psi_dy2_1, d2psi_dxdy_1 = _hessian_m3_1(r, phi, q)
                d2psi_dx2_2, d2psi_dy2_2, d2psi_dxdy_2 = _hessian_m3_1(
                    r, phi + np.pi / 2, 1 / q
                )
                f_xx = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dx2_1
                        + (1 / q) * np.sin(m * phi_m) * d2psi_dy2_2
                    )
                )
                f_yy = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dy2_1
                        + (1 / q) * np.sin(m * phi_m) * d2psi_dx2_2
                    )
                )
                f_xy = (
                    a_m
                    * np.sqrt(q)
                    * (
                        np.cos(m * phi_m) * d2psi_dxdy_1
                        - (1 / q) * np.sin(m * phi_m) * d2psi_dxdy_2
                    )
                )

            elif (m % 2) == 0:  # for m=4, will also work for any even m
                phi_ell = np.angle(q * r * np.cos(phi) + 1j * r * np.sin(phi))
                R = np.sqrt(q * (r * np.cos(phi)) ** 2 + (r * np.sin(phi)) ** 2 / q)

                delta_r = a_m * np.cos(m * (phi_ell - phi_m)) * r / R
                f_xx = np.sin(phi) ** 2 * delta_r / r
                f_yy = np.cos(phi) ** 2 * delta_r / r
                f_xy = -np.sin(phi) * np.cos(phi) * delta_r / r

            else:
                raise ValueError(
                    "Implementation of multipoles perturbation for general axis ratio q only available for m=1, m=3 or m=4."
                )

        return f_xx, f_xy, f_xy, f_yy


def _phi_ell(phi, q):
    return (
        phi
        - np.arctan2(np.sin(phi), np.cos(phi))
        + np.arctan2(np.sin(phi), q * np.cos(phi))
    )


def _G_m_1(m, phi, q):
    return np.cos(m * np.arctan2(np.sin(phi), q * np.cos(phi))) / np.sqrt(
        q**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2
    )


def _F_m1_1_hat(phi, q):
    term1 = np.cos(phi) * (
        q * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        - (np.log(2) * (1 + q) / 2 - (1 - q**2) * (1 + np.log(2) / 4))
    )
    term2 = 2 * np.sin(phi) * (phi - _phi_ell(phi, q))
    return -(term1 + term2) / (2 * (1 - q**2))


def _F_m1_1_hat_derivative(phi, q):
    term1 = -np.cos(phi) * q * 2 * (q**2 - 1) * np.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * np.cos(2 * phi)
    ) + np.sin(phi) * (
        -q * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi))
        + np.log(2) * (1 + q) / 2
        - (1 - q**2) * (1 + np.log(2) / 4)
    )
    term2 = 2 * np.cos(phi) * (phi - _phi_ell(phi, q)) + 2 * np.sin(phi) * (
        1 - q / (q**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2)
    )
    return -(term1 + term2) / (2 * (1 - q**2))


def _potential_m1_1(r, phi, q, r_E):
    lambda_m1 = 2 / (1 + q)
    return r * _F_m1_1_hat(phi, q) + lambda_m1 / 2 * r * np.log(r / r_E) * np.cos(phi)


def _alpha_m1_1(r, phi, q, r_E):
    lambda_m1 = 2 / (1 + q)
    f_phi = _F_m1_1_hat(phi, q)
    df_dphi = _F_m1_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * np.cos(phi)
        - df_dphi * np.sin(phi)
        + lambda_m1 / 2 * (np.log(r / r_E) + np.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * np.sin(phi)
        + df_dphi * np.cos(phi)
        + lambda_m1 / 2 * np.cos(phi) * np.sin(phi)
    )
    return alpha_x, alpha_y


def _hessian_m1_1(r, phi, q):
    lambda_m1 = 2 / (1 + q)
    G_m1_1 = _G_m_1(1, phi, q)
    d2psi_dx2 = (np.sin(phi) ** 2 * G_m1_1 + lambda_m1 / 2 * np.cos(phi)) / r
    d2psi_dy2 = (np.cos(phi) ** 2 * G_m1_1 - lambda_m1 / 2 * np.cos(phi)) / r
    d2psi_dxdy = (-np.cos(phi) * np.sin(phi) * G_m1_1 + lambda_m1 / 2 * np.sin(phi)) / r
    return d2psi_dx2, d2psi_dy2, d2psi_dxdy


def _A_3_1(q):
    return (
        np.log(2) * (1 + q) ** 2
        - 2 * (1 - q) * (1 + q) ** 2 * (1 + np.log(2) / 4)
        + (1 - q**2) ** 2 / 4
    )


def _F_m3_1_hat(phi, q):
    term1 = np.cos(phi) * (
        q * (3 + q**2) * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi)) - _A_3_1(q)
    )
    term2 = 2 * np.sin(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q))
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _F_m3_1_hat_derivative(phi, q):
    term1 = -np.cos(phi) * q * (3 + q**2) * 2 * (q**2 - 1) * np.sin(2 * phi) / (
        1 + q**2 + (q**2 - 1) * np.cos(2 * phi)
    ) + np.sin(phi) * (
        -q * (3 + q**2) * np.log(1 + q**2 + (q**2 - 1) * np.cos(2 * phi)) + _A_3_1(q)
    )
    term2 = 2 * np.cos(phi) * (1 + 3 * q**2) * (phi - _phi_ell(phi, q)) + 2 * np.sin(
        phi
    ) * (1 + 3 * q**2) * (1 - q / (q**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2))
    return (term1 + term2) / (2 * (1 - q**2) ** 2)


def _potential_m3_1(r, phi, q, r_E):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    return r * _F_m3_1_hat(phi, q) + lambda_m3 / 2 * r * np.log(r / r_E) * np.cos(phi)


def _alpha_m3_1(r, phi, q, r_E):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    f_phi = _F_m3_1_hat(phi, q)
    df_dphi = _F_m3_1_hat_derivative(phi, q)
    alpha_x = (
        f_phi * np.cos(phi)
        - df_dphi * np.sin(phi)
        + lambda_m3 / 2 * (np.log(r / r_E) + np.cos(phi) ** 2)
    )
    alpha_y = (
        f_phi * np.sin(phi)
        + df_dphi * np.cos(phi)
        + lambda_m3 / 2 * np.cos(phi) * np.sin(phi)
    )
    return alpha_x, alpha_y


def _hessian_m3_1(r, phi, q):
    lambda_m3 = -2 * (1 - q) / (1 + q) ** 2
    G_m3_1 = _G_m_1(3, phi, q)
    d2psi_dx2 = (np.sin(phi) ** 2 * G_m3_1 + lambda_m3 / 2 * np.cos(phi)) / r
    d2psi_dy2 = (np.cos(phi) ** 2 * G_m3_1 - lambda_m3 / 2 * np.cos(phi)) / r
    d2psi_dxdy = (-np.cos(phi) * np.sin(phi) * G_m3_1 + lambda_m3 / 2 * np.sin(phi)) / r
    return d2psi_dx2, d2psi_dy2, d2psi_dxdy


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
