__author__ = "Maverick-Oh"

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.epl import EPL
from lenstronomy.LensModel.Profiles.multipole import Multipole, EllipticalMultipole


__all__ = ["EPL_BOXYDISKY_ELL", "EPL_BOXYDISKY"]


class EPL_BOXYDISKY_ELL(LensProfileBase):
    """ " EPL (Elliptical Power Law) mass profile combined with an elliptical multipole
    with m=4, so that it's either purely boxy or disky with EPL's axis and multipole's
    axis aligned (exact for general axis ratio q).

    Read the documentation of lenstronomy.LensModel.Profiles.epl and
    lenstronomy.LensModel.Profiles.multipole for details.

    :param theta_E: Einstein radius
    :param gamma: negative power-law slope of the 3D mass distributions
    :param e1: eccentricity. For details, read
        lenstronomy.Util.param_util.phi_q2_ellipticity document.
    :param e2: eccentricity. For details, read
        lenstronomy.Util.param_util.phi_q2_ellipticity document.
    :param center_x: center of distortion
    :param center_y: center of distortion
    :param a4_a: Strength of the deviation of multipole order 4 of the elliptical
        isodensity contours, which is translated into the multipole strength from the
        MULTIPOLE_ELL class through a rescaling by theta_E. Profile is disky when a4_a>0
        and boxy when a4_a<0.
    """

    param_names = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y", "a4_a"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "a4_a": -0.1,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "a4_a": +0.1,
    }

    def __init__(self):
        self._epl = EPL()
        self._multipole = EllipticalMultipole()
        self._m = int(4)
        super(EPL_BOXYDISKY_ELL, self).__init__()

    def _param_split(self, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """This function splits the keyword arguments for the EPL and multipole
        profiles.

        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a4_a: amplitude of the multipole mass profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: the keyword arguments for the joint profile
        """

        phi, q = param_util.ellipticity2phi_q(e1, e2)
        kwargs_epl = {
            "theta_E": theta_E,
            "gamma": gamma,
            "e1": e1,
            "e2": e2,
            "center_x": center_x,
            "center_y": center_y,
        }
        kwargs_multipole = {
            "m": self._m,
            "a_m": a4_a * theta_E,
            "phi_m": phi,
            "q": q,
            "center_x": center_x,
            "center_y": center_y,
        }

        return kwargs_epl, kwargs_multipole

    def function(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_epl = self._epl.function(x, y, **kwargs_epl)
        f_multipole = self._multipole.function(x, y, **kwargs_multipole)
        return f_epl + f_multipole

    def derivatives(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_x_epl, f_y_epl = self._epl.derivatives(x, y, **kwargs_epl)
        f_x_multipole, f_y_multipole = self._multipole.derivatives(
            x, y, **kwargs_multipole
        )
        f_x = f_x_epl + f_x_multipole
        f_y = f_y_epl + f_y_multipole
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_xx_epl, f_xy_epl, f_yx_epl, f_yy_epl = self._epl.hessian(x, y, **kwargs_epl)
        (
            f_xx_multipole,
            f_xy_multipole,
            f_yx_multipole,
            f_yy_multipole,
        ) = self._multipole.hessian(x, y, **kwargs_multipole)
        f_xx = f_xx_epl + f_xx_multipole
        f_xy = f_xy_epl + f_xy_multipole
        f_yx = f_yx_epl + f_yx_multipole
        f_yy = f_yy_epl + f_yy_multipole
        return f_xx, f_xy, f_yx, f_yy


class EPL_BOXYDISKY(LensProfileBase):
    """ " EPL (Elliptical Power Law) mass profile combined with a circular multipole with
    m=4, so that it's either purely boxy or disky with EPL's axis and multipole's axis
    aligned (exact for axis ratio q=1 only).

    Reference to the implementation: https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.127V/abstract

    Read the documentation of lenstronomy.LensModel.Profiles.epl and lenstronomy.LensModel.Profiles.multipole for details.

    :param theta_E: Einstein radius
    :param gamma: negative power-law slope of the 3D mass distributions
    :param e1: eccentricity. For details, read lenstronomy.Util.param_util.phi_q2_ellipticity document.
    :param e2: eccentricity. For details, read lenstronomy.Util.param_util.phi_q2_ellipticity document.
    :param center_x: center of distortion
    :param center_y: center of distortion
    :param a4_a: Strength of the deviation of multipole order 4 of the elliptical isodensity contours,
     which is translated into the multipole strength from the MULTIPOLE class through a rescaling by theta_E / sqrt(q).
     Profile is disky when a4_a>0 and boxy when a4_a<0.
    """

    param_names = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y", "a4_a"]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "a4_a": -0.1,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "a4_a": +0.1,
    }

    def __init__(self):
        self._epl = EPL()
        self._multipole = Multipole()
        self._m = int(4)
        super(EPL_BOXYDISKY, self).__init__()

    def _param_split(self, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """This function splits the keyword arguments for the EPL and multipole
        profiles.

        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a4_a: amplitude of the multipole mass profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: the keyword arguments for the joint profile
        """

        phi, q = param_util.ellipticity2phi_q(e1, e2)
        rescale_am = theta_E / np.sqrt(q)
        kwargs_epl = {
            "theta_E": theta_E,
            "gamma": gamma,
            "e1": e1,
            "e2": e2,
            "center_x": center_x,
            "center_y": center_y,
        }
        kwargs_multipole = {
            "m": self._m,
            "a_m": a4_a * rescale_am,
            "phi_m": phi,
            "center_x": center_x,
            "center_y": center_y,
        }

        return kwargs_epl, kwargs_multipole

    def function(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_epl = self._epl.function(x, y, **kwargs_epl)
        f_multipole = self._multipole.function(x, y, **kwargs_multipole)
        return f_epl + f_multipole

    def derivatives(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_x_epl, f_y_epl = self._epl.derivatives(x, y, **kwargs_epl)
        f_x_multipole, f_y_multipole = self._multipole.derivatives(
            x, y, **kwargs_multipole
        )
        f_x = f_x_epl + f_x_multipole
        f_y = f_y_epl + f_y_multipole
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, a4_a, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param a4_a: multipole strength. The profile becomes disky when a4_a>0 and boxy when a4_a<0
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        kwargs_epl, kwargs_multipole = self._param_split(
            theta_E, gamma, e1, e2, a4_a, center_x=center_x, center_y=center_y
        )
        f_xx_epl, f_xy_epl, f_yx_epl, f_yy_epl = self._epl.hessian(x, y, **kwargs_epl)
        (
            f_xx_multipole,
            f_xy_multipole,
            f_yx_multipole,
            f_yy_multipole,
        ) = self._multipole.hessian(x, y, **kwargs_multipole)
        f_xx = f_xx_epl + f_xx_multipole
        f_xy = f_xy_epl + f_xy_multipole
        f_yx = f_yx_epl + f_yx_multipole
        f_yy = f_yy_epl + f_yy_multipole
        return f_xx, f_xy, f_yx, f_yy
