
import numpy as np

__all__ = ["LineProfile"]

class LineProfile(object):
    """
    Horizontal line segment class.
    Parameters:
    start_x: ra-coordiinate of start of line
    start_y: dec-coordinate of start of line
    length: length of line (arcseconds)
    width: width of line (arcseconds), line centered at start_x, start_y in perpendicular direction
    amp: surface brightness of line
    angle: angle of jet to the horizontal (degrees, 0 = constant RA)
    """

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
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: keyword arguments of profile
        :return: surface brightness, raise as definition is not defined
        """
        
        ang = -np.deg2rad(angle)
        out = (np.cos(ang)*(start_x - x) + np.sin(ang)*(start_y - y), np.cos(ang)*(start_y - y) - np.sin(ang)*(start_x - x))
        
        return amp * ((out[0]) > 0) * ((out[0]) < length) * (abs((out[1])) < width/2)




    def light_3d(self, *args, **kwargs):
        """

        :param r: 3d radius
        :param kwargs:  keyword arguments of profile
        :return: 3d light profile, raise as definition is not defined
        """
        raise ValueError("light_3d definition not defined in the light profile.")