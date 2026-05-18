__author__ = "dangilman"

import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.epl import EPL
from lenstronomy.LensModel.Profiles.multipole import Multipole, EllipticalMultipole

__all__ = ["EPL_MULTIPOLE_M3M4", "EPL_MULTIPOLE_M3M4_ELL"]


class EPL_MULTIPOLE_M3M4_ELL(LensProfileBase):
    """EPL (Elliptical Power Law) mass profile combined with two elliptical multipole
    terms of order m=3 and m=4 (exact for general axis ratio q).

    See also documentation of EPL_BOXYDIKSY CLASS, lenstronomy.LensModel.Profiles.epl
    and lenstronomy.LensModel.Profiles.multipole for details.
    """

    param_names = [
        "theta_E",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
        "a3_a",
        "delta_phi_m3",
        "a4_a",
        "delta_phi_m4",
    ]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "a3_a": -0.2,
        "delta_phi_m3": -np.pi / 6,
        "a4_a": -0.2,
        "delta_phi_m4": -np.pi / 8,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "a3_a": 0.2,
        "delta_phi_m3": np.pi / 6,
        "a4_a": 0.2,
        "delta_phi_m4": np.pi / 8,
    }

    def __init__(self):
        self._epl = EPL()
        self._multipole = EllipticalMultipole()
        super(EPL_MULTIPOLE_M3M4_ELL, self).__init__()

    def _param_split(
        self,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """This function splits the keyword arguments for the EPL and multipole
        profiles.

        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
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
        kwargs_multipole_m3 = {
            "m": 3,
            "a_m": a3_a * theta_E,
            "phi_m": phi + delta_phi_m3,
            "q": q,
            "center_x": center_x,
            "center_y": center_y,
            "r_E": theta_E,
        }
        kwargs_multipole_m4 = {
            "m": 4,
            "a_m": a4_a * theta_E,
            "phi_m": phi + delta_phi_m4,
            "q": q,
            "center_x": center_x,
            "center_y": center_y,
        }

        return kwargs_epl, kwargs_multipole_m3, kwargs_multipole_m4

    def function(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the gravitational potential in units of theta_E^2.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: lensing potential.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_epl = self._epl.function(x, y, **kwargs_epl)
        f_multipole = self._multipole.function(x, y, **kwargs_multipole3)
        f_multipole += self._multipole.function(x, y, **kwargs_multipole4)
        return f_epl + f_multipole

    def derivatives(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the derivatives of the potential (deflection angles)in units of
        theta_E.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: alpha_x, alpha_y.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_x_epl, f_y_epl = self._epl.derivatives(x, y, **kwargs_epl)
        f_x_multipole3, f_y_multipole3 = self._multipole.derivatives(
            x, y, **kwargs_multipole3
        )
        f_x_multipole4, f_y_multipole4 = self._multipole.derivatives(
            x, y, **kwargs_multipole4
        )
        f_x = f_x_epl + f_x_multipole3 + f_x_multipole4
        f_y = f_y_epl + f_y_multipole3 + f_y_multipole4
        return f_x, f_y

    def hessian(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the components of the hessian matrix (second derivatives of the
        potential)

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE_ELL profile by a scaling
            theta_E
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: f_xx, f_xy, f_yx, f_yy.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_xx_epl, f_xy_epl, f_yx_epl, f_yy_epl = self._epl.hessian(x, y, **kwargs_epl)
        (
            f_xx_multipole3,
            f_xy_multipole3,
            f_yx_multipole3,
            f_yy_multipole3,
        ) = self._multipole.hessian(x, y, **kwargs_multipole3)
        (
            f_xx_multipole4,
            f_xy_multipole4,
            f_yx_multipole4,
            f_yy_multipole4,
        ) = self._multipole.hessian(x, y, **kwargs_multipole4)
        f_xx = f_xx_epl + f_xx_multipole3 + f_xx_multipole4
        f_xy = f_xy_epl + f_xy_multipole3 + f_xy_multipole4
        f_yx = f_yx_epl + f_yx_multipole3 + f_yx_multipole4
        f_yy = f_yy_epl + f_yy_multipole3 + f_yy_multipole4
        return f_xx, f_xy, f_yx, f_yy


class EPL_MULTIPOLE_M3M4(LensProfileBase):
    """EPL (Elliptical Power Law) mass profile combined with two circular multipole
    terms of order m=3 and m=4 (exact for axis ratio =1).

    Reference to the implementation: https://ui.adsabs.harvard.edu/abs/2022A%26A...659A.127V/abstract

    See also documentation of EPL_BOXYDIKSY CLASS, lenstronomy.LensModel.Profiles.epl and
    lenstronomy.LensModel.Profiles.multipole for details.
    """

    param_names = [
        "theta_E",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
        "a3_a",
        "delta_phi_m3",
        "a4_a",
        "delta_phi_m4",
    ]
    lower_limit_default = {
        "theta_E": 0,
        "gamma": 1.5,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "a3_a": -0.2,
        "delta_phi_m3": -np.pi / 6,
        "a4_a": -0.2,
        "delta_phi_m4": -np.pi / 8,
    }
    upper_limit_default = {
        "theta_E": 100,
        "gamma": 2.5,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "a3_a": 0.2,
        "delta_phi_m3": np.pi / 6,
        "a4_a": 0.2,
        "delta_phi_m4": np.pi / 8,
    }

    def __init__(self):
        self._epl = EPL()
        self._multipole = Multipole()
        super(EPL_MULTIPOLE_M3M4, self).__init__()

    def _param_split(
        self,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """This function splits the keyword arguments for the EPL and multipole
        profiles.

        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
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
        kwargs_multipole_m3 = {
            "m": 3,
            "a_m": a3_a * rescale_am,
            "phi_m": phi + delta_phi_m3,
            "center_x": center_x,
            "center_y": center_y,
        }
        kwargs_multipole_m4 = {
            "m": 4,
            "a_m": a4_a * rescale_am,
            "phi_m": phi + delta_phi_m4,
            "center_x": center_x,
            "center_y": center_y,
        }

        return kwargs_epl, kwargs_multipole_m3, kwargs_multipole_m4

    def function(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the gravitational potential in units of theta_E^2.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: lensing potential.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_epl = self._epl.function(x, y, **kwargs_epl)
        f_multipole = self._multipole.function(x, y, **kwargs_multipole3)
        f_multipole += self._multipole.function(x, y, **kwargs_multipole4)
        return f_epl + f_multipole

    def derivatives(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the derivatives of the potential (deflection angles)in units of
        theta_E.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: alpha_x, alpha_y.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_x_epl, f_y_epl = self._epl.derivatives(x, y, **kwargs_epl)
        f_x_multipole3, f_y_multipole3 = self._multipole.derivatives(
            x, y, **kwargs_multipole3
        )
        f_x_multipole4, f_y_multipole4 = self._multipole.derivatives(
            x, y, **kwargs_multipole4
        )
        f_x = f_x_epl + f_x_multipole3 + f_x_multipole4
        f_y = f_y_epl + f_y_multipole3 + f_y_multipole4
        return f_x, f_y

    def hessian(
        self,
        x,
        y,
        theta_E,
        gamma,
        e1,
        e2,
        a3_a,
        delta_phi_m3,
        a4_a,
        delta_phi_m4,
        center_x=0,
        center_y=0,
    ):
        """Computes the components of the hessian matrix (second derivatives of the
        potential)

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: log-slope of EPL mass profile
        :param e1: ellipticity of EPL profile (along 1st axis)
        :param e2: ellipticity of EPL profile (along 2nd axis)
        :param a3_a: amplitude of the m=3 multiple deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m3: orientation of the m=3 profile relative to the position
            angle of the EPL profile
        :param a4_a: amplitude of the m=4 multipole deviation from pure elliptical shape
            related to the physical amplitude of the MULTIPOLE profile by a scaling
            theta_E / sqrt(q)
        :param delta_phi_m4: orientation of the m=4 profile relative to the position
            angle of the EPL profile
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: f_xx, f_xy, f_yx, f_yy.
        """
        kwargs_epl, kwargs_multipole3, kwargs_multipole4 = self._param_split(
            theta_E,
            gamma,
            e1,
            e2,
            a3_a,
            delta_phi_m3,
            a4_a,
            delta_phi_m4,
            center_x=center_x,
            center_y=center_y,
        )
        f_xx_epl, f_xy_epl, f_yx_epl, f_yy_epl = self._epl.hessian(x, y, **kwargs_epl)
        (
            f_xx_multipole3,
            f_xy_multipole3,
            f_yx_multipole3,
            f_yy_multipole3,
        ) = self._multipole.hessian(x, y, **kwargs_multipole3)
        (
            f_xx_multipole4,
            f_xy_multipole4,
            f_yx_multipole4,
            f_yy_multipole4,
        ) = self._multipole.hessian(x, y, **kwargs_multipole4)
        f_xx = f_xx_epl + f_xx_multipole3 + f_xx_multipole4
        f_xy = f_xy_epl + f_xy_multipole3 + f_xy_multipole4
        f_yx = f_yx_epl + f_yx_multipole3 + f_yx_multipole4
        f_yy = f_yy_epl + f_yy_multipole3 + f_yy_multipole4
        return f_xx, f_xy, f_yx, f_yy
