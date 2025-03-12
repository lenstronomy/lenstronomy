__author__ = "gipagano"

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["ConstMag"]


class ConstMag(LensProfileBase):
    """This class implements the macromodel potential of Diego et al.

    `ConstMag` assumes constant radial and tangential magnification components.

    The lensing potential is given by:
    .. math::
        \\psi(\\theta_x, \\theta_y) = \\frac{\\kappa}{2}(\\theta_x^2 + \\theta_y^2) + \\frac{\\gamma_1}{2}(\\theta_x^2 - \\theta_y^2) - \\gamma_2(\\theta_x \\theta_y)
    where:
    :math:`\\theta_x` and :math:`\\theta_y` are angular coordinates in the image plane,
    :math:`\\gamma_1` and :math:`\\gamma_2` are the horizontal and vertical components of shear, respectively, and
    :math:`\\kappa` is the convergence.


    :math:`\\kappa` and :math:`\\gamma` depend on the parity. A negative parity means the image becomes inverted compared to the source.

    For positive parity (:math:`\\text{parity} = +1`):
    .. math::
        \\gamma = \\frac{1}{2} \\left(\\frac{1}{\\mu_t} - \\frac{1}{\\mu_r}\\right)
        \\kappa = 1 - \\gamma - \\frac{1}{\\mu_r}

    For negative parity (:math:`\\text{parity} = -1`):
    .. math::
        \\gamma = \\frac{1}{2} \\left(\\frac{1}{\\mu_t} + \\frac{1}{\\mu_r}\\right)
        \\kappa = 1 - \\gamma + \\frac{1}{\\mu_r}
    where :math:`mu_r` and :math:`mu_t` are the radial and tangental components of magnification, respectively.

    The shear components are calculated as:
    .. math::
        \\gamma_1 = \\gamma\\cos{2\\phi_G}
        \\gamma_2 = -\\gamma\\sin{2\\phi_G}
    where :math:`\\phi_G` is the shear orientation angle relative to the x-axis.

    For a detailed derivation see <https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf>`
    _
    Convergence and shear are computed according to `Diego2018 <arXiv:1706.10281v2>`
    """

    param_names = ["center_x", "center_y", "mu_r", "mu_t", "parity", "phi_G"]
    lower_limit_default = {
        "center_x": -100,
        "center_y": -100,
        "mu_r": 1,
        "mu_t": 1000,
        "parity": -1,
        "phi_G": 0.0,
    }
    upper_limit_default = {
        "center_x": 100,
        "center_y": 100,
        "mu_r": 1,
        "mu_t": 1000,
        "parity": 1,
        "phi_G": np.pi,
    }

    def function(self, x, y, mu_r, mu_t, parity, phi_G, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification
        :param mu_t: tangential magnification
        :param parity: parity side of the macromodel. Either +1 (positive parity) or -1 (negative parity)
        :param phi_G: shear orientation angle (relative to the x-axis)
        :return: lensing potential
        """

        # positive parity case
        if parity == 1:
            gamma = (1.0 / mu_t - 1.0 / mu_r) * 0.5
            kappa = 1 - gamma - 1.0 / mu_r

        # negative parity case
        elif parity == -1:
            gamma = (1.0 / mu_t + 1.0 / mu_r) * 0.5
            kappa = 1 - gamma + 1.0 / mu_r
        else:
            raise ValueError(
                "%f is not a valid value for the parity of the macromodel. Choose either +1 or -1."
                % parity
            )

        # compute the shear along the x and y directions, rotate the vector in the opposite direction than the reference frame (compare with util.rotate)
        gamma1, gamma2 = gamma * np.cos(2 * phi_G), -gamma * np.sin(2 * phi_G)

        x_shift = x - center_x
        y_shift = y - center_y
        f_ = (
            1.0 / 2.0 * kappa * (x_shift * x_shift + y_shift * y_shift)
            + 1.0 / 2.0 * gamma1 * (x_shift * x_shift - y_shift * y_shift)
            - gamma2 * x_shift * y_shift
        )

        return f_

    def derivatives(self, x, y, mu_r, mu_t, parity, phi_G, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification
        :param mu_t: tangential magnification
        :param parity: parity of the side of the macromodel. Either +1 (positive parity) or -1 (negative parity)
        :param phi_G: shear orientation angle (relative to the x-axis)
        :return: deflection angle (in angles)
        """

        # positive parity case
        if parity == 1:
            gamma = (1.0 / mu_t - 1.0 / mu_r) * 0.5
            kappa = 1 - gamma - 1.0 / mu_r

        # negative parity case
        elif parity == -1:
            gamma = (1.0 / mu_t + 1.0 / mu_r) * 0.5
            kappa = 1 - gamma + 1.0 / mu_r
        else:
            raise ValueError(
                "%f is not a valid value for the parity of the macromodel. Choose either +1 or -1."
                % parity
            )

        # compute the shear along the x and y directions, rotate the vector in the opposite direction than the reference frame (compare with util.rotate)
        gamma1, gamma2 = gamma * np.cos(2 * phi_G), -gamma * np.sin(2 * phi_G)

        x_shift = x - center_x
        y_shift = y - center_y
        f_x = (kappa + gamma1) * x_shift - gamma2 * y_shift
        f_y = (kappa - gamma1) * y_shift - gamma2 * x_shift

        return f_x, f_y

    def hessian(self, x, y, mu_r, mu_t, parity, phi_G, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param mu_r: radial magnification
        :param mu_t: tangential magnification
        :param parity: parity of the side of the macromodel. Either +1 (positive parity) or -1 (negative parity)
        :param phi_G: shear orientation angle (relative to the x-axis)
        :return: hessian matrix (in angles)
        """

        # positive parity case
        if parity == 1:
            gamma = (1.0 / mu_t - 1.0 / mu_r) * 0.5
            kappa = 1 - gamma - 1.0 / mu_r

        # negative parity case
        elif parity == -1:
            gamma = (1.0 / mu_t + 1.0 / mu_r) * 0.5
            kappa = 1 - gamma + 1.0 / mu_r
        else:
            raise ValueError(
                "%f is not a valid value for the parity of the macromodel. Choose either +1 or -1."
                % parity
            )

        # compute the shear along the x and y directions, rotate the vector in the opposite direction than the reference frame (compare with util.rotate)
        gamma1, gamma2 = gamma * np.cos(2 * phi_G), -gamma * np.sin(2 * phi_G)

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = -gamma2

        return f_xx, f_xy, f_xy, f_yy
