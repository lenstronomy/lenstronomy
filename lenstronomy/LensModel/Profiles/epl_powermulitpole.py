__author__ = "eckerl"

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.epl import EPL

__all__ = ["EPL_PMULTIPOL", "EPL_PMULTIPOL_QPHI"]


class EPL_PMultipol(LensProfileBase):
    """This class contains a EPL+PLMultipole contribution over e1,e2.
    The PLMultipole is an extention to the parametrization of Chu et al.(2013) (https://arxiv.org/abs/1302.5482). The equation is Eq. (8) from Nightingale et al. (2023) (https://arxiv.org/abs/2209.10566)
    It scales the contribution of the multipoles with the same power-law as the EPL.  It is defined with a prefactor 1/2 and the einstein radius theta_e in order to achieve k = theta_E / 2theta as the m=0
    contribution (Reason stated in Chu et al.(2013) (https://arxiv.org/abs/1302.5482) Eq. (3)).


    theta_E : float, Einstein radius
    gamma : float, power-law slope
    e1: eccentricity component
    e2: eccentricity component
    m : int, multipole order, m>=2
    k_m : float, multipole strength and sqrt (a_m^2+b_m^2) of old definition.
    phi_m : float, multipole orientation in radian
    center_x: x-position
    center_y: y-position
    """

    param_names = [
        "theta_E",
        "gamma",
        "e1",
        "e2",
        "m",
        "k_m",
        "phi_m",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "m": 0,
        "k_m": 0,
        "phi_m": -np.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "m": 100,
        "k_m": 100,
        "phi_m": np.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self._EPL = EPL()
        super(EPL_PMultipol, self).__init__()

    def function(
        self, x, y, theta_E, gamma, e1, e2, m, k_m, phi_m, center_x=0, center_y=0
    ):
        """
        Lensing potential of PLmultipole contribution (for 1 component with m>=2)+EPL.
        The equation for PLMultipol is Eq. (8) from Nightingale et al. (2023) (https://arxiv.org/abs/2209.10566)



        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        f_multi = (
            theta_E ** (gamma - 1)
            * k_m
            / ((3 - gamma) ** 2 - m**2)
            * r ** (3 - gamma)
            * np.cos(m * (phi - phi_m))
        )
        f_epl = self._EPL.function(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        f_ = f_epl + f_multi
        return f_

    def derivatives(
        self, x, y, theta_E, gamma, e1, e2, m, k_m, phi_m, center_x=0, center_y=0
    ):
        """
        Deflection of a multipole contribution (for 1 component with m>=2)
        This uses an extention to the parametrization of Chu et al.(2013) (https://arxiv.org/abs/1302.5482). The equation is Eq. (8) from Nightingale et al. (2023) (https://arxiv.org/abs/2209.10566)
        It scales the contribution of the multipoles with the same power-law as the EPL.  It is defined with a prefactor 1/2 and the einstein radius theta_e in order to achieve k = theta_E / 2theta as the m=0
        contribution (Reason stated in Chu et al.(2013) (https://arxiv.org/abs/1302.5482) Eq. (3)).

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m**2)
        f_x_multi = pre_factor * (
            np.cos(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m))
            + np.sin(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m))
        )
        f_y_multi = pre_factor * (
            np.sin(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m))
            - np.cos(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m))
        )

        f_x_epl, f_y_epl = self._EPL.derivatives(
            x, y, theta_E, gamma, e1, e2, center_x, center_y
        )

        f_x = f_x_epl + f_x_multi
        f_y = f_y_epl + f_y_multi

        return f_x, f_y

    def hessian(
        self, x, y, theta_E, gamma, e1, e2, m, k_m, phi_m, center_x=0, center_y=0
    ):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m**2)

        term1_xx = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.cos(phi)
            * (
                np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                + np.sin(phi) * m * np.sin(m * (phi - phi_m))
            )
        )
        term2_xx = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.sin(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m**2)
                + (2 - gamma)
                * np.cos(phi)
                * np.sin(phi)
                * m
                * np.sin(m * (phi - phi_m))
            )
        )
        f_xx_multi = term1_xx + term2_xx

        term1_yy = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.sin(phi)
            * (
                np.sin(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                - np.cos(phi) * m * np.sin(m * (phi - phi_m))
            )
        )

        term2_yy = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.cos(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m**2)
                - (2 - gamma)
                * np.cos(phi)
                * np.sin(phi)
                * m
                * np.sin(m * (phi - phi_m))
            )
        )

        f_yy_multi = term1_yy + term2_yy

        # Term calculations
        term1_xy = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.sin(phi)
            * (
                np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                + np.sin(phi) * m * np.sin(m * (phi - phi_m))
            )
        )

        term2_xy = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.sin(phi)
                * np.cos(phi)
                * np.cos(m * (phi - phi_m))
                * (-(3 - gamma) + m**2)
                - (2 - gamma) * np.cos(phi) ** 2 * m * np.sin(m * (phi - phi_m))
            )
        )

        f_xy_multi = term1_xy + term2_xy

        f_xx_epl, f_xy_epl, f_xy_epl, f_yy_epl = self._EPL.hessian(
            x, y, theta_E, gamma, e1, e2, center_x, center_y
        )

        f_xx = f_xx_epl + f_xx_multi
        f_xy = f_xy_epl + f_xy_multi
        f_yy = f_yy_epl + f_yy_multi

        return f_xx, f_xy, f_xy, f_yy

    def density_lens(self, *args, **kwargs):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity. (optional definition)

        .. math::
            \\kappa(x, y) = \\int_{-\\infty}^{\\infty} \\rho(x, y, z) dz

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "density_lens definition is not defined in the profile you want to execute."
        )

    def mass_3d_lens(self, *args, **kwargs):
        """Mass enclosed within a 3D sphere or radius r given a lens parameterization
        with angular units. The input parameter are identical as for the derivatives
        definition. (optional definition)

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "mass_3d_lens definition is not defined in the profile you want to execute."
        )

    def mass_2d_lens(self, *args, **kwargs):
        """Two-dimensional enclosed mass at radius r (optional definition)

        .. math::
            M_{2d}(R) = \\int_{0}^{R} \\rho_{2d}(r) 2\\pi r dr

        with :math:`\\rho_{2d}(r)` is the density_2d_lens() definition

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "mass_2d_lens definition is not defined in the profiel you want to execute."
        )

    def set_static(self, **kwargs):
        """Pre-computes certain computations that do only relate to the lens model
        parameters and not to the specific position where to evaluate the lens model.

        :param kwargs: lens model parameters
        :return: no return, for certain lens model some private self variables are
            initiated
        """
        pass

    def set_dynamic(self):
        """

        :return: no return, deletes pre-computed variables for certain lens models
        """
        pass


class EPL_PMultipol_qphi(LensProfileBase):
    """This class contains a EPL+PLMultipole contribution over q and phi instead of e1,e2.
    The PLMultipole is an extention to the parametrization of Chu et al.(2013) (https://arxiv.org/abs/1302.5482). The equation is Eq. (8) from Nightingale et al. (2023) (https://arxiv.org/abs/2209.10566)
    It scales the contribution of the multipoles with the same power-law as the EPL.  It is defined with a prefactor 1/2 and the einstein radius theta_e in order to achieve k = theta_E / 2theta as the m=0
    contribution (Reason stated in Chu et al.(2013) (https://arxiv.org/abs/1302.5482) Eq. (3)).


    theta_E : float, Einstein radius
    gamma : float, power-law slope
    q: axis ratio
    phi: position angle
    m : int, multipole order, m>=2
    k_m : float, multipole strength and sqrt (a_m^2+b_m^2) of old definition.
    phi_m : float, multipole orientation in radian
    center_x: x-position
    center_y: y-position
    """

    param_names = [
        "theta_E",
        "gamma",
        "q",
        "m",
        "k_m",
        "phi_m",
        "phi",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "q": 0,
        "m": 0,
        "k_m": 0,
        "phi_m": -np.pi,
        "phi": -np.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "q": 1,
        "m": 100,
        "k_m": 100,
        "phi_m": np.pi,
        "phi": np.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self._epl = EPL()

    def function(
        self, x, y, theta_E, gamma, q, m, phi, k_m, phi_m, center_x=0, center_y=0
    ):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        f_multi = (
            theta_E ** (gamma - 1)
            * k_m
            / ((3 - gamma) ** 2 - m**2)
            * r ** (3 - gamma)
            * np.cos(m * (phi - phi_m))
        )
        f_epl = self._EPL.function(x, y, theta_E, gamma, e1, e2, center_x, center_y)
        f_ = f_epl + f_multi
        return f_

    def derivatives(
        self, x, y, theta_E, gamma, q, phi, k_m, m, phi_m, center_x=0, center_y=0
    ):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m**2)
        f_x_multi = pre_factor * (
            np.cos(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m))
            + np.sin(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m))
        )
        f_y_multi = pre_factor * (
            np.sin(phi) * r ** (2 - gamma) * np.cos(m * (phi - phi_m))
            - np.cos(phi) * r ** (1 - gamma) * m * np.sin(m * (phi - phi_m))
        )

        f_x_epl, f_y_epl = self._EPL.derivatives(
            x, y, theta_E, gamma, e1, e2, center_x, center_y
        )

        f_x = f_x_epl + f_x_multi
        f_y = f_y_epl + f_y_multi

        return f_x, f_y

    def hessian(
        self, x, y, theta_E, gamma, q, phi, k_m, m, phi_m, center_x=0, center_y=0
    ):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :param phi: position angle
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 0.000001)
        pre_factor = theta_E ** (gamma - 1) * k_m / ((3 - gamma) ** 2 - m**2)

        term1_xx = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.cos(phi)
            * (
                np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                + np.sin(phi) * m * np.sin(m * (phi - phi_m))
            )
        )
        term2_xx = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.sin(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m**2)
                + (2 - gamma)
                * np.cos(phi)
                * np.sin(phi)
                * m
                * np.sin(m * (phi - phi_m))
            )
        )
        f_xx_multi = term1_xx + term2_xx

        term1_yy = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.sin(phi)
            * (
                np.sin(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                - np.cos(phi) * m * np.sin(m * (phi - phi_m))
            )
        )

        term2_yy = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.cos(phi) ** 2 * np.cos(m * (phi - phi_m)) * ((3 - gamma) - m**2)
                - (2 - gamma)
                * np.cos(phi)
                * np.sin(phi)
                * m
                * np.sin(m * (phi - phi_m))
            )
        )

        f_yy_multi = term1_yy + term2_yy

        # Term calculations
        term1_xy = (
            pre_factor
            * r ** (1 - gamma)
            * (2 - gamma)
            * np.sin(phi)
            * (
                np.cos(phi) * (3 - gamma) * np.cos(m * (phi - phi_m))
                + np.sin(phi) * m * np.sin(m * (phi - phi_m))
            )
        )

        term2_xy = (
            pre_factor
            * r ** (1 - gamma)
            * (
                np.sin(phi)
                * np.cos(phi)
                * np.cos(m * (phi - phi_m))
                * (-(3 - gamma) + m**2)
                - (2 - gamma) * np.cos(phi) ** 2 * m * np.sin(m * (phi - phi_m))
            )
        )

        f_xy_multi = term1_xy + term2_xy

        f_xx_epl, f_xy_epl, f_xy_epl, f_yy_epl = self._EPL.hessian(
            x, y, theta_E, gamma, e1, e2, center_x, center_y
        )

        f_xx = f_xx_epl + f_xx_multi
        f_xy = f_xy_epl + f_xy_multi
        f_yy = f_yy_epl + f_yy_multi

        return f_xx, f_xy, f_xy, f_yy

    def density_lens(self, *args, **kwargs):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity. (optional definition)

        .. math::
            \\kappa(x, y) = \\int_{-\\infty}^{\\infty} \\rho(x, y, z) dz

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "density_lens definition is not defined in the profile you want to execute."
        )

    def mass_3d_lens(self, *args, **kwargs):
        """Mass enclosed within a 3D sphere or radius r given a lens parameterization
        with angular units. The input parameter are identical as for the derivatives
        definition. (optional definition)

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "mass_3d_lens definition is not defined in the profile you want to execute."
        )

    def mass_2d_lens(self, *args, **kwargs):
        """Two-dimensional enclosed mass at radius r (optional definition)

        .. math::
            M_{2d}(R) = \\int_{0}^{R} \\rho_{2d}(r) 2\\pi r dr

        with :math:`\\rho_{2d}(r)` is the density_2d_lens() definition

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param kwargs: keywords of the profile
        :return: raise as definition is not defined
        """
        raise ValueError(
            "mass_2d_lens definition is not defined in the profiel you want to execute."
        )

    def set_static(self, **kwargs):
        """Pre-computes certain computations that do only relate to the lens model
        parameters and not to the specific position where to evaluate the lens model.

        :param kwargs: lens model parameters
        :return: no return, for certain lens model some private self variables are
            initiated
        """
        pass

    def set_dynamic(self):
        """

        :return: no return, deletes pre-computed variables for certain lens models
        """
        pass
