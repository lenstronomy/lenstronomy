__author__ = "sibirrer"
# this file contains a class to make a gaussian

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["Gaussian"]


class Gaussian(LensProfileBase):
    """This class contains functions to evaluate a Gaussian function and calculates its
    derivative and hessian matrix."""

    param_names = ["amp", "sigma_x", "sigma_y", "center_x", "center_y"]
    lower_limit_default = {"amp": 0, "sigma": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"amp": 100, "sigma": 100, "center_x": 100, "center_y": 100}

    def function(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns Gaussian."""
        c = amp / (2 * np.pi * sigma_x * sigma_y)
        delta_x = x - center_x
        delta_y = y - center_y
        exponent = -((delta_x / sigma_x) ** 2 + (delta_y / sigma_y) ** 2) / 2.0
        return c * np.exp(exponent)

    def derivatives(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function."""
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        return f_ * (center_x - x) / sigma_x**2, f_ * (center_y - y) / sigma_y**2

    def hessian(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
        d^f/dy^2."""
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        f_xx = f_ * ((-1.0 / sigma_x**2) + (center_x - x) ** 2 / sigma_x**4)
        f_yy = f_ * ((-1.0 / sigma_y**2) + (center_y - y) ** 2 / sigma_y**4)
        f_xy = f_ * (center_x - x) / sigma_x**2 * (center_y - y) / sigma_y**2
        return f_xx, f_xy, f_xy, f_yy
