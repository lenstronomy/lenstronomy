__author__ = "abstractlegwear"

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.constants as const
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

__all__ = ["CosmicString"]

class CosmicString(LensProfileBase):
    """
    Cosmic string profile, where the string is a straight 1d topological defect.
    Local metric of string is flat, but global spacetime is conical.
    This results in the total radians around a string being 2π - ∆ø, where ∆ø is the deficit angle.
    ∆ø = 8πGµ/c^2, where µ is the linear density in kg/m.
    The string deflects light from a source by the constant angle 4πGµ, resulting in two images with 0 magnification or shear.
    This model is simplified such that the string is completely flat on image plane (no rotation towards/away from observer) and it is currently of infinite length.
    """
    
    param_names = ["alpha", "alpha_hat", "theta", "center_x", "center_y"]
    lower_limit_default = {"alpha": 0, "alpha_hat": 0, "theta": 0, "center_x": -100, "center_y": -100}
    upper_limit_default = {"alpha": 1, "alpha_hat": 1, "theta": np.pi, "center_x": 100, "center_y": 100}
    
    # not sure what the limits for alpha/alpha_hat should be
    
    def __init__(self):
        # not sure if I need __init__
        1
    
    def function(self, x, y, alpha, alpha_hat, theta, center_x = 0, center_y = 0):
        """
        Lensing potential
        
        :param x: image's angular position (normally in arc seconds)
        :param y: image's angular position (normally in arc seconds)
        :param alpha: reduced deflection angle
        :param theta: angle of string (east to north) with a limit of [0, pi] since [0, 2pi] is redundant
        :param center_x: center of string (in angular units)
        :param center_y: center of string (in angular units)
        :return: lensing potential
        """
        return alpha * abs((y-center_y)*np.cos(theta)-(x-center_x)*np.sin(theta))
    
    def derivatives(self, x, y, alpha, alpha_hat, theta, center_x = 0, center_y = 0):
        """
        Reduced deflection angle
        
        :param x: image's angular position (normally in arc seconds)
        :param y: image's angular position (normally in arc seconds)
        :param alpha: reduced deflection angle
        :param alpha_hat: physical deflection angle
        :param theta: angle of string (east to north) with a limit of [0, pi] since [0, 2pi] is redundant
        :param center_x: center of string (in angular units)
        :param center_y: center of string (in angular units)
        :param length: to be added, length of string
        :return: alpha_x, alpha_y
        """
        x_ = x - center_x
        y_ = y - center_y
        
        # rotate ccw by pi/2 - theta to make string vertical
        
        x__, y__ = util.rotate(x_, y_, np.pi/2 - theta)
        
        # then find whether on right or left of string, on string, or outside of einstein strip
        # s is the einstein strip's bound for the image (not for the source!)
        # if source is outside the strip, is it just image with 0 distortion or nothing?
        
        s = 2 * alpha_hat + alpha  # removing distance dependency for now, it's just an angle
        
        if isinstance(x, int) or isinstance(x, float):
            if x__ > s or x__ < -1 * s or x__ == 0:
                side = 0
            elif x__ > 0:
                side = 1
            elif x__ < 0:
                side = -1
        else:
            side = np.zeros_like(x)
            for i in range(x__.size):
                if x__[i] > 0:
                    side[i] = 1
                elif x__[i] < 0:
                    side[i] = -1          
            
        # side = 1 if right or -1 if left, 0 if outside of strip or on string (not sure if this is right)
        
        psi = abs(theta - np.pi / 2)  # look in notes for psi
        
        alpha_x = side * alpha * np.cos(psi)
        
        # alpha_x is always + if on right and - if on left
        # alpha_y changes based on side and theta
        
        if theta < 90:  
            alpha_y = -1 * side * alpha * np.sin(psi)
        elif theta > 90: 
            alpha_y = side * alpha * np.sin(psi)
        elif theta == 90: # when string is vertical alpha_y = 0
            alpha_y = 0
            
        return alpha_x, alpha_y
    
    def hessian(self, x, y, alpha, alpha_hat, theta, center_x = 0, center_y = 0):
        """
        Hessian of lensing potential
        :return: f_xx, f_xy, f_yx, f_yy
        """
        if isinstance(x, int) or isinstance(x, float):
            f_xx, f_xy, f_yx, f_yy = 0, 0, 0, 0
        else:
            f_xx, f_xy, f_yx, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        return f_xx, f_xy, f_yx, f_yy
