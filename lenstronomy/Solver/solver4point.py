__author__ = 'sibirrer'

import scipy.optimize
import numpy as np
from astrofunc.LensingProfiles.spep import SPEP
from astrofunc.LensingProfiles.spemd import SPEMD
from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE
from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
from astrofunc.LensingProfiles.composite_sersic_nfw import CompositeSersicNFW
import astrofunc.util as util


class SolverProfile(object):
    """
    class to solve multidimensional non-linear equations for 4 point image
    """
    def __init__(self, lens_model='SPEP'):
        if lens_model == 'SPEP':
            self.lens = SPEP()
        elif lens_model == 'SPEMD':
            self.lens = SPEMD()
        else:
            raise ValueError('lens model %s not valid for solver type "PROFILE" ' % lens_model)

    def F(self, x, x_cat, y_cat, a, gamma):
        """

        :param x: array of parameters
        :return:
        """
        [theta_E, e1, e2, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, theta_E, gamma, q, phi_G, center_x, center_y)
        y = np.zeros(6)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha1[0] - alpha1[2]
        y[2] = alpha1[0] - alpha1[3]
        y[3] = alpha2[0] - alpha2[1]
        y[4] = alpha2[0] - alpha2[2]
        y[5] = alpha2[0] - alpha2[3]
        return y - a

    def solve(self, init, x_cat, y_cat, a, gamma):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, gamma), xtol=1.49012e-08, factor=0.1)
        return x


class SolverNFWProfile(object):
    """
    class to solve multidimensional non-linear equations for 4 point image
    """
    def __init__(self, lens_model='NFW_ELLIPSE'):
        if lens_model == 'NFW_ELLIPSE':
            self.lens = NFW_ELLIPSE()
        else:
            raise ValueError('lens model %s not valid for solver type "NFW_PROFILE" ' % lens_model)

    def F(self, x, x_cat, y_cat, a, Rs):
        """

        :param x: array of parameters
        :return:
        """
        [theta_Rs, e1, e2, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, theta_Rs, Rs, q, phi_G, center_x, center_y)
        y = np.zeros(6)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha1[0] - alpha1[2]
        y[2] = alpha1[0] - alpha1[3]
        y[3] = alpha2[0] - alpha2[1]
        y[4] = alpha2[0] - alpha2[2]
        y[5] = alpha2[0] - alpha2[3]
        return y - a

    def solve(self, init, x_cat, y_cat, a, Rs):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, Rs), xtol=1.49012e-08, factor=0.1)
        return x


class SolverComposite(object):
    """
    class to solve multidimensional non-linear equations for 4 point image
    """
    def __init__(self, lens_model='COMPOSITE'):
        if lens_model == 'COMPOSITE':
            self.lens = CompositeSersicNFW()
        else:
            raise ValueError('lens model %s not valid for solver type "COMPOSITE" ' % lens_model)

    def F(self, x, x_cat, y_cat, a, mass_light, Rs, n_sersic, r_eff, q_s, phi_G_s):
        """

        :param x: array of parameters
        :return:
        """
        [theta_E, e1, e2, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, theta_E, mass_light, Rs, q, phi_G, n_sersic, r_eff, q_s, phi_G_s, center_x, center_y)
        y = np.zeros(6)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha1[0] - alpha1[2]
        y[2] = alpha1[0] - alpha1[3]
        y[3] = alpha2[0] - alpha2[1]
        y[4] = alpha2[0] - alpha2[2]
        y[5] = alpha2[0] - alpha2[3]
        return y - a

    def solve(self, init, x_cat, y_cat, a, mass_light, Rs, n_sersic, r_eff, q_s, phi_G_s):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, mass_light, Rs, n_sersic, r_eff, q_s, phi_G_s), xtol=1.49012e-08, factor=0.1)
        return x


class SolverShapelets(object):

    def __init__(self):
        self.shapelets = CartShapelets()

    def F(self, x, x_cat, y_cat, a, beta, center_x, center_y):
        [c00, c10, c01, c20, c11, c02] = x
        coeffs = [0, c10, c01, c20, c11, c02]
        alpha1, alpha2 = self.shapelets.derivatives(x_cat, y_cat, coeffs, beta, center_x, center_y)
        y = np.zeros(6)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha1[0] - alpha1[2]
        y[2] = alpha1[0] - alpha1[3]
        y[3] = alpha2[0] - alpha2[1]
        y[4] = alpha2[0] - alpha2[2]
        y[5] = alpha2[0] - alpha2[3]
        return y - a

    def solve(self, init, x_cat, y_cat, a, beta, center_x, center_y):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, beta, center_x, center_y), xtol=1.49012e-10)#, factor=0.1)
        return x


class Constraints(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, solver_type='PROFILE', lens_model='SPEP'):
        if solver_type == 'PROFILE':
            self.solver = SolverProfile(lens_model)
        elif solver_type == 'NFW_PROFILE':
            self.solver = SolverNFWProfile(lens_model)
        elif solver_type == 'COMPOSITE':
            self.solver = SolverComposite(lens_model)
        elif solver_type == 'SHAPELETS':
            self.solver = SolverShapelets()
        elif solver_type == 'NONE':
            pass
        else:
            raise ValueError('invalid solver type!')

    def _subtract_constraint(self, x_cat, y_cat, x_sub, y_sub):
        """

        :param x_cat:
        :param y_cat:
        :param x_sub:
        :param y_sub:
        :return:
        """
        a = np.zeros(6)
        a[0] = x_cat[0] - x_cat[1] - x_sub[0] + x_sub[1]
        a[1] = x_cat[0] - x_cat[2] - x_sub[0] + x_sub[2]
        a[2] = x_cat[0] - x_cat[3] - x_sub[0] + x_sub[3]
        a[3] = y_cat[0] - y_cat[1] - y_sub[0] + y_sub[1]
        a[4] = y_cat[0] - y_cat[2] - y_sub[0] + y_sub[2]
        a[5] = y_cat[0] - y_cat[3] - y_sub[0] + y_sub[3]
        return a

    def get_param(self, x_cat, y_cat, x_sub, y_sub, init, kwargs):
        """

        :param x_cat:
        :param y_cat:
        :param x_sub:
        :param y_sub:
        :return:
        """
        a = self._subtract_constraint(x_cat, y_cat, x_sub, y_sub)
        x = self.solver.solve(init, x_cat, y_cat, a, **kwargs)
        return x