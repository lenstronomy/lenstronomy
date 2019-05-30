__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
import numpy as np


class Shear(object):
    """
    new class for external shear e1, e2 expression
    """
    param_names = ['e1', 'e2', 'ra_0', 'dec_0']
    lower_limit_default = {'e1': -0.5, 'e2': -0.5, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'e1': 0.5, 'e2': 0.5, 'ra_0': 100, 'dec_0': 100}

    def function(self, x, y, e1, e2, ra_0=0, dec_0=0):
        # change to polar coordinates
        psi_ext, gamma_ext = param_util.ellipticity2phi_gamma(e1, e2)
        r, phi = param_util.cart2polar(x-ra_0, y-dec_0)
        f_ = 1./2 * gamma_ext * r**2 * np.cos(2*(phi - psi_ext))
        #x_ = x - ra_0
        #y_ = y - dec_0
        #f_ = 1/2. * (e1 * x_ * x_ + 2 * e2 * x_ * y_ - e1 * y_ * y_)
        return f_

    def derivatives(self, x, y, e1, e2, ra_0=0, dec_0=0):
        # rotation angle
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = e1*x_ + e2*y_
        f_y = +e2*x_ - e1*y_
        return f_x, f_y

    def hessian(self, x, y, e1, e2, ra_0=0, dec_0=0):
        gamma1 = e1
        gamma2 = e2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy
