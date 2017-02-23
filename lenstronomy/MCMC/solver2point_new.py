__author__ = 'sibirrer'

import scipy.optimize
import numpy as np
from astrofunc.LensingProfiles.spep import SPEP
from astrofunc.LensingProfiles.spemd import SPEMD
from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
import astrofunc.util as util


class SolverSPEP2_new(object):
    """
    class to solve multidimensional non-linear equations for 2 point image
    """
    def __init__(self):
        self.spep = SPEP()

    def F(self, x, x_cat, y_cat, a, center_x, center_y, e1, e2):
        """

        :param x: array of parameters
        :return:
        """
        [theta_E, gamma] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.spep.derivatives(x_cat, y_cat, theta_E, gamma, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, center_x, center_y, e1, e2):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, center_x, center_y, e1, e2), xtol=1.49012e-08, factor=0.1)
        return x


class SolverSPEMD2_new(object):
    """
    class to solve multidimensional non-linear equations for 4 point image
    """
    def __init__(self):
        self.spemd = SPEMD()

    def F(self, x, x_cat, y_cat, a, center_x, center_y, e1, e2):
        """

        :param x: array of parameters
        :return:
        """
        [theta_E, gamma] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.spemd.derivatives(x_cat, y_cat, theta_E, gamma, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, theta_E, gamma, e1, e2):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, theta_E, gamma, e1, e2), xtol=1.49012e-08, factor=0.1)
        return x


class SolverShapelets2_new(object):

    def __init__(self):
        self.shapelets = CartShapelets()

    def F(self, x, x_cat, y_cat, a, beta, center_x, center_y):
        [c10, c01] = x
        coeffs = [0, c10, c01]
        alpha1, alpha2 = self.shapelets.derivatives(x_cat, y_cat, coeffs, beta, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, beta, center_x, center_y):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, beta, center_x, center_y), xtol=1.49012e-10)#, factor=0.1)
        return x


class Constraints2_new(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, solver_type='SPEP'):
        if solver_type == 'SPEP':
            self.solver = SolverSPEP2_new()
        elif solver_type == 'SPEMD':
            self.solver = SolverSPEMD2_new()
        elif solver_type == 'SHAPELETS':
            self.solver = SolverShapelets2_new()
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
        a = np.zeros(2)
        a[0] = x_cat[0] - x_cat[1] - x_sub[0] + x_sub[1]
        a[1] = y_cat[0] - y_cat[1] - y_sub[0] + y_sub[1]
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