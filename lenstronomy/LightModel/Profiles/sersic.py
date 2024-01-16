__author__ = "sibirrer"

#  this file contains a class to make a Sersic profile

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
import lenstronomy.Util.param_util as param_util
import numpy as np
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class Sersic(SersicUtil):
    """This class contains functions to evaluate a spherical Sersic function.

    .. math::
        I(R) = I_0 \\exp \\left[ -b_n \\left[(R/R_{\\rm Sersic})^{\\frac{1}{n}}-1\\right]\\right]

    with :math:`I_0 = amp`
    and
    with :math:`b_{n}\\approx 1.999n-0.327`
    """

    param_names = ["amp", "R_sersic", "n_sersic", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "center_x": 100,
        "center_y": 100,
    }

    def function(
        self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=1000.0
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """
        R = self.get_distance_from_center(
            x, y, e1=0, e2=0, center_x=center_x, center_y=center_y
        )
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


@export
class SersicElliptic(SersicUtil):
    """This class contains functions to evaluate an elliptical Sersic function.

    .. math::

        I(R) = I_0 \\exp \\left[ -b_n \\left[(R/R_{\\rm Sersic})^{\\frac{1}{n}}-1\\right]\\right]

    with :math:`I_0 = amp`,
    :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`
    and
    with :math:`b_{n}\\approx 1.999n-0.327`
    """

    param_names = ["amp", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        n_sersic,
        e1,
        e2,
        center_x=0,
        center_y=0,
        max_R_frac=1000.0,
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter e1
        :param e2: eccentricity parameter e2
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        R_sersic = np.maximum(0, R_sersic)
        R = self.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


@export
class SersicElliptic_qPhi(SersicUtil):
    """This class is the same as SersicElliptic except sampling over q and phi instead
    of e1 and e2."""

    param_names = ["amp", "R_sersic", "n_sersic", "q", "phi", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "R_sersic": 0,
        "n_sersic": 0.5,
        "q": 0,
        "phi": -np.pi,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "R_sersic": 100,
        "n_sersic": 8,
        "q": 1.0,
        "phi": np.pi,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, *args, **kwargs):
        self._sersic_e1e2 = SersicElliptic(*args, **kwargs)

    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        n_sersic,
        q,
        phi,
        center_x=0,
        center_y=0,
        max_R_frac=100.0,
    ):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param n_sersic: Sersic index
        :param q: axis ratio
        :param phi: position angle (radians)
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return self._sersic_e1e2.function(
            x, y, amp, R_sersic, n_sersic, e1, e2, center_x, center_y, max_R_frac
        )


@export
class CoreSersic(SersicUtil):
    """This class contains the Core-Sersic function introduced by e.g. Trujillo et al.
    2004.

    .. math::

        I(R) = I' \\left[1 + (R_b/R)^{\\alpha} \\right]^{\\gamma / \\alpha}
        \\exp \\left{ -b_n \\left[(R^{\\alpha} + R_b^{\\alpha})/R_e^{\\alpha}  \\right]^{1 / (n\\alpha)}  \\right}

    with

    .. math::
        I' = I_b 2^{-\\gamma/ \\alpha} \\exp \\left[b_n 2^{1 / (n\\alpha)} (R_b/R_e)^{1/n}  \\right]

    where :math:`I_b` is the intensity at the break radius and :math:`R = \\sqrt{q \\theta^2_x + \\theta^2_y/q}`.
    """

    param_names = [
        "amp",
        "R_sersic",
        "Rb",
        "n_sersic",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "amp": 0,
        "Rb": 0,
        "n_sersic": 0.5,
        "gamma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "Rb": 100,
        "n_sersic": 8,
        "gamma": 10,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def function(
        self,
        x,
        y,
        amp,
        R_sersic,
        Rb,
        n_sersic,
        gamma,
        e1,
        e2,
        center_x=0,
        center_y=0,
        alpha=3.0,
        max_R_frac=1000.0,
    ):
        """
        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param Rb: "break" core radius
        :param n_sersic: Sersic index
        :param gamma: inner power-law exponent
        :param e1: eccentricity parameter e1
        :param e2: eccentricity parameter e2
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param alpha: sharpness of the transition between the cusp and the outer Sersic profile (float)
        :param max_R_frac: maximum window outside which the mass is zeroed, in units of R_sersic (float)
        :return: Cored Sersic profile value at (x, y)
        """
        # TODO: max_R_frac not implemented
        R_ = self.get_distance_from_center(x, y, e1, e2, center_x, center_y)
        R = self._R_stable(R_)
        bn = self.b_n(n_sersic)
        result = (
            amp
            * (1 + (Rb / R) ** alpha) ** (gamma / alpha)
            * np.exp(
                -bn
                * (
                    ((R**alpha + Rb**alpha) / R_sersic**alpha)
                    ** (1.0 / (alpha * n_sersic))
                    - 1.0
                )
            )
        )
        return np.nan_to_num(result)
