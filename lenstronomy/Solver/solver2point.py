__author__ = 'sibirrer'

import scipy.optimize
import numpy as np
from astrofunc.LensingProfiles.spep import SPEP
from astrofunc.LensingProfiles.spemd import SPEMD
from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
from astrofunc.LensingProfiles.shapelet_pot import PolarShapelets
from astrofunc.LensingProfiles.external_shear import ExternalShear
from astrofunc.LensingProfiles.nfw_ellipse import NFW_ELLIPSE
import astrofunc.util as util


class SolverCenter2(object):
    """
    class to solve multidimensional non-linear equations for 2 point image
    """
    def __init__(self, lens_model='SPEP'):
        if lens_model == 'SPEP':
            self.lens = SPEP()
        elif lens_model == 'SPEMD':
            self.lens = SPEMD()
        else:
            raise ValueError('lens model %s not valid for solver type CENTER!' % lens_model)

    def F(self, x, x_cat, y_cat, a, theta_E, gamma, e1, e2):
        """

        :param x: array of parameters
        :return:
        """
        [center_x, center_y] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, theta_E, gamma, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, theta_E, gamma, e1, e2):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, theta_E, gamma, e1, e2), xtol=1.49012e-08, factor=0.1)
        return x


class SolverEllipse2(object):
    """
    class to solve multidimensional non-linear equations for 2 point image
    """
    def __init__(self, lens_model='SPEP'):
        if lens_model == 'SPEP':
            self.lens = SPEP()
        elif lens_model == 'SPEMD':
            self.lens = SPEMD()
        else:
            raise ValueError('lens model %s not valid for solver type ELLIPSE!' % lens_model)

    def F(self, x, x_cat, y_cat, a, theta_E, gamma, center_x, center_y):
        """

        :param x: array of parameters
        :return:
        """
        [e1, e2] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, theta_E, gamma, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, theta_E, gamma, center_x, center_y):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, theta_E, gamma, center_x, center_y), xtol=1.49012e-08, factor=0.1)
        return x


class SolverNFWCenter2(object):
    """
    class to solve multidimensional non-linear equations for 2 point image
    """
    def __init__(self, lens_model='NFW_ELLIPSE'):
        if lens_model == 'NFW_ELLIPSE':
            self.lens = NFW_ELLIPSE()
        else:
            raise ValueError('lens model %s not valid for solver type NFW_CENTER!' % lens_model)

    def F(self, x, x_cat, y_cat, a, Rs, theta_Rs, e1, e2):
        """

        :param x: array of parameters
        :return:
        """
        [center_x, center_y] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, Rs, theta_Rs, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, theta_Rs, Rs, e1, e2):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, Rs, theta_Rs, e1, e2), xtol=1.49012e-08, factor=0.1)
        return x


class SolverNFWEllipse2(object):
    """
    class to solve multidimensional non-linear equations for 2 point image
    """
    def __init__(self, lens_model='NFW_ELLIPSE'):
        if lens_model == 'NFW_ELLIPSE':
            self.lens = NFW_ELLIPSE()
        else:
            raise ValueError('lens model %s not valid for solver type NFW_CENTER!' % lens_model)

    def F(self, x, x_cat, y_cat, a, Rs, theta_Rs, center_x, center_y):
        """

        :param x: array of parameters
        :return:
        """
        [e1, e2] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        alpha1, alpha2 = self.lens.derivatives(x_cat, y_cat, Rs, theta_Rs, q, phi_G, center_x, center_y)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, theta_Rs, Rs, center_x, center_y):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a, Rs, theta_Rs, center_x, center_y), xtol=1.49012e-08, factor=0.1)
        return x


class SolverShapelets2(object):

    def __init__(self, lens_model='SHAPELETS_CART'):
        if lens_model == 'SHAPELETS_CART':
            self.shapelets = CartShapelets()
        elif lens_model == 'SHAPELETS_POLAR':
            self.shapelets = PolarShapelets()
        else:
            raise ValueError('lens model %s not valid for solver type "SHAPELETS" ' % lens_model)

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


class SolverShear2(object):

    def __init__(self, lens_model='EXTERNAL_SHEAR'):
        if lens_model == 'EXTERNAL_SHEAR':
            self.shear = ExternalShear()
        else:
            raise ValueError('lens model %s not valid for solver type SHAPELET!' % lens_model)

    def F(self, x, x_cat, y_cat, a):
        [e1, e2] = x
        alpha1, alpha2 = self.shear.derivatives(x_cat, y_cat, e1=e1, e2=e2)
        y = np.zeros(2)
        y[0] = alpha1[0] - alpha1[1]
        y[1] = alpha2[0] - alpha2[1]
        return y - a

    def solve(self, init, x_cat, y_cat, a, **kwargs):
        x = scipy.optimize.fsolve(self.F, init, args=(x_cat, y_cat, a), xtol=1.49012e-10)#, factor=0.1)
        return x


class Constraints2(object):
    """
    class to make the constraints for the solver
    """
    def __init__(self, solver_type='CENTER', lens_model='SPEP'):
        if solver_type == 'CENTER':
            self.solver = SolverCenter2(lens_model)
        elif solver_type == 'ELLIPSE':
            self.solver = SolverEllipse2(lens_model)
        elif solver_type == 'NFW_CENTER':
            self.solver = SolverNFWCenter2(lens_model)
        elif solver_type == 'NFW_ELLIPSE':
            self.solver = SolverNFWEllipse2(lens_model)
        elif solver_type == 'SHAPELETS':
            self.solver = SolverShapelets2(lens_model)
        elif solver_type == 'SHEAR':
            self.solver = SolverShear2(lens_model)
        elif solver_type == 'NONE':
            pass
        else:
            raise ValueError('invalid solver type: %s !' % solver_type)

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
