__author__ = 'TheoDuboscq'

#  this file contains a class to make a Sersic profile with flexion. Created to speed up computation with the minimal flexion model, not very useful otherwise.

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class SersicEllipticFlexed(SersicUtil):
    """
    this class contains functions to evaluate an elliptical Sersic function which contains flexion

    .. math::

        I(z) = I_{sersic}(z + az + bz^* + c^*z^2 + 2czz^* + d(z^*)^2)

    with :math:`I_{sersic}` the profile of an elliptical sersic, and 
    :math:`z = x + iy`, and x, y the coordinates of the point in which we are about to calculate the profile.

    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y', 'a', 'b1', 'b2', 'c1', 'c2', 'd1', 'd2']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100, 
                           'a': -0.5, 'b1': -0.5, 'b2': -0.5, 'c1': -0.5, 'c2': -0.5, 'd1': -0.5, 'd2': -0.5}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100, 
                           'a': 0.5, 'b1': 0.5, 'b2': 0.5, 'c1': 0.5, 'c2': 0.5, 'd1': 0.5, 'd2': 0.5}

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0, max_R_frac=100.0, a=0, b1=0, b2=0, c1=0, c2=0, d1=0, d2=0):
        """

        :param x: x coordinate of the point in which the profile is about to be evaluated
        :param y: y coordinate of the point in which the profile is about to be evaluated
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: half light radius (either semi-major axis or product average of semi-major and semi-minor axis)
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :param a: real number a in the above formula
        :param b1: real part of the complex number b in the above formula
        :param b2: imaginary part of the complex number b in the above formula
        :param c1: real part of the complex number c in the above formula
        :param c2: imaginary part of the complex number c in the above formula
        :param d1: real part of the complex number d in the above formula
        :param d2: imaginary part of the complex number d in the above formula
        :return: Sersic flexion profile value at (x, y)
        """

        R_sersic = np.maximum(0, R_sersic)
        b, c, d = b1 + 1j*b2, c1 + 1j*c2, d1 + 1j*d2
        z, center = x + 1j*y, center_x + 1j*center_y
        Z = z + a*z + b*z.conjugate() + c.conjugate()*z**2 + 2*c*z*z.conjugate() + d*z.conjugate()**2
        center_shift = center + a*center + b*center.conjugate() + c.conjugate()*center**2 + 2*c*center*center.conjugate() + d*center.conjugate()**2
        X, Y = Z.real, Z.imag
        center_shift_x, center_shift_y = center_shift.real, center_shift.imag
        R = self.get_distance_from_center(X, Y, e1, e2, center_shift_x, center_shift_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result

