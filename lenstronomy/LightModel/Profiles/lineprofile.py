import numpy as np


__all__ = ["LineProfile"]


class LineProfile(object):
    """Horizontal line segment class."""

    param_names = ["amp", "angle", "length", "width", "start_x", "start_y"]
    lower_limit_default = {
        "amp": 0,
        "angle": -180,
        "length": 0.01,
        "width": 0.01,
        "start_x": -100,
        "start_y": -100,
    }
    upper_limit_default = {
        "amp": 10,
        "angle": 180,
        "length": 10,
        "width": 5,
        "start_x": 100,
        "start_y": 100,
    }

    def __init__(self):
        pass

    def function(self, x, y, amp, angle, length, width, start_x=0, start_y=0):
        """Surface brightness per angular unit.

        :param x: x-coordinate on sky
        :param y: y-coordinate on sky
        :param amp: constant surface brightness of line
        :param angle: angle of line to the horizontal (degrees)
        :param length: length of line (arcseconds)
        :param width: width of line (arcseconds), line width extends symmetrically
        :param start_x: ra coordinate of start of line
        :param start_y: dec-coordinate of start of line
        :return: surface brightness, raise as definition is not defined
        """
        ang = -np.deg2rad(angle)
        x_ = np.cos(ang) * (start_x - x) + np.sin(ang) * (start_y - y)
        y_ = np.cos(ang) * (start_y - y) - np.sin(ang) * (start_x - x)
        return amp * (x_ > 0) * (x_ < length) * (abs(y_) < width / 2)

    def total_flux(self, amp, angle, length, width, start_x=0, start_y=0):
        """Integrated flux of the profile.

        :param amp: constant surface brightness of line
        :param angle: angle of line to the horizontal (degrees)
        :param length: length of line (arcseconds)
        :param width: width of line (arcseconds), line width extends symmetrically
        :param start_x: ra coordinate of start of line
        :param start_y: dec-coordinate of start of line
        :return: total flux
        """
        return amp * length * width

    def light_3d(self, *args, **kwargs):
        """

        :param r: 3d radius
        :param kwargs:  keyword arguments of profile
        :return: 3d light profile, raise as definition is not defined
        """
        raise ValueError("light_3d definition not defined in the light profile.")
