__author__ = 'lynevdv'


import numpy as np

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

class Multipole(LensProfileBase):
    """
    This class contains a multipole contribution (for 1 component with m>=2)
    m : int, multipole order
    a_m : float, multipole strength
    phi_m : float, multipole orientation in radian
    """
    param_names = ['m', 'a_m', 'phi_m', 'center_x', 'center_y']
    lower_limit_default = {'m': 2,'a_m':0, 'phi_m':-np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'m': 100,'a_m':100, 'phi_m':np.pi, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, m, a_m, phi_m, center_x=0, center_y=0):
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_ = r*a_m /(1-m**2) *np.cos(m*(phi-phi_m))
        return f_
    def derivatives(self,x,y, m, a_m, phi_m, center_x=0, center_y=0):
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        f_x = np.cos(phi)*a_m/(1-m**2) *np.cos(m*(phi-phi_m)) + np.sin(phi)*m*a_m/(1-m**2)*np.sin(m*(phi-phi_m))
        f_y = np.sin(phi)*a_m/(1-m**2) *np.cos(m*(phi-phi_m)) - np.cos(phi)*m*a_m/(1-m**2)*np.sin(m*(phi-phi_m))
        return f_x,f_y
    def hessian(self,x,y, m, a_m, phi_m, center_x=0, center_y=0):
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        if isinstance(r, int) or isinstance(r, float):
            r = max(0.000001, r)
        else:
            r[r < 0.000001] = 0.000001
        f_xx = 1./r * np.sin(phi)**2 * a_m *np.cos(m*(phi-phi_m))
        f_yy = 1./r * np.cos(phi)**2 * a_m *np.cos(m*(phi-phi_m))
        f_xy = -1./r * a_m * np.cos(phi) * np.sin(phi) * np.cos(m*(phi-phi_m))
        return f_xx, f_yy, f_xy