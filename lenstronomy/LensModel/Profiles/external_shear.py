__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
import numpy as np

class ExternalShear_old(object):
    """
    class to deal with external shear. We do not include external convergence at this stage
    """

    def function(self, x, y, gamma_ext, psi_ext):
        # change to polar coordinates
        theta, phi = param_util.cart2polar(x, y)
        f_ = 1./2 * gamma_ext * theta**2 * np.cos(2*(phi - psi_ext))
        return f_

    def derivatives(self, x, y, gamma_ext, psi_ext):
        # rotation angle
        sin_phi = np.sin(psi_ext)
        cos_phi = np.cos(psi_ext)
        x_ = cos_phi*x + sin_phi*y
        y_ = -sin_phi*x + cos_phi*y

        f_x_prim = gamma_ext * y_
        f_y_prim = gamma_ext * x_
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, gamma_ext, psi_ext):
        # rotation angle
        sin_phi = np.sin(psi_ext)
        cos_phi = np.cos(psi_ext)
        x_ = cos_phi*x + sin_phi*y
        y_ = -sin_phi*x + cos_phi*y
        f_xx_prim = np.zeros_like(x_)
        f_yy_prim = np.zeros_like(x_)
        f_xy_prim = gamma_ext * np.ones_like(x_)
        kappa = 1./2 * (f_xx_prim + f_yy_prim)
        gamma1_value = 1./2 * (f_xx_prim -f_yy_prim)
        gamma2_value = f_xy_prim
        # rotate back
        gamma1 = np.cos(2*psi_ext)*gamma1_value-np.sin(2*psi_ext)*gamma2_value
        gamma2 = +np.sin(2*psi_ext)*gamma1_value+np.cos(2*psi_ext)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy


class ExternalShear(object):
    """
    new class for external shear e1, e2 expression
    """

    def function(self, x, y, e1, e2):
        # change to polar coordinates
        psi_ext, gamma_ext = param_util.ellipticity2phi_gamma(e1, e2)
        theta, phi = param_util.cart2polar(x, y)
        f_ = 1./2 * gamma_ext * theta**2 * np.cos(2*(phi - psi_ext))
        return f_

    def derivatives(self, x, y, e1, e2):
        # rotation angle
        f_x = e1*x + e2*y
        f_y = +e2*x - e1*y
        return f_x, f_y

    def hessian(self, x, y, e1, e2):
        gamma1 = e1
        gamma2 = e2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy
