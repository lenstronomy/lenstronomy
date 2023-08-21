__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.Util import util

__all__ = ['CSE', 'CSEMajorAxis', 'CSEMajorAxisSet', 'CSEProductAvg', 'CSEProductAvgSet']


class CSE(LensProfileBase):
    """
    Cored steep ellipsoid (CSE)
    :param axis: 'major' or 'product_avg' ; whether to evaluate corresponding to r= major axis or r= sqrt(ab)
    source:
    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{x^2 + \\frac{y^2}{q^2}}

    """
    param_names = ['A', 's', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'A': -1000, 's': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'A': 1000, 's': 10000, 'e1': 0.5, 'e2': 0.5, 'center_x': -100, 'center_y': -100}

    def __init__(self, axis='product_avg'):
        if axis == 'major':
            self.major_axis_model = CSEMajorAxis()
        elif axis == 'product_avg':
            self.major_axis_model = CSEProductAvg()
        else:
            raise ValueError("axis must be set to 'major' or 'product_avg'. Input is %s ." % axis)
        super(CSE, self).__init__()

    def function(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: lensing potential
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        f_ = self.major_axis_model.function(x__, y__, a, s, q)
        return f_

    def derivatives(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: deflection in x- and y-direction
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        f__x, f__y = self.major_axis_model.derivatives(x__, y__, a, s, q)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        return f_x, f_y

    def hessian(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        f__xx, f__xy, __, f__yy = self.major_axis_model.hessian(x__, y__, a, s, q)

        # rotate back
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_q) * gamma1__ - np.sin(2 * phi_q) * gamma2__
        gamma2 = +np.sin(2 * phi_q) * gamma1__ + np.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return f_xx, f_xy, f_xy, f_yy


class CSEMajorAxis(LensProfileBase):
    """
    Cored steep ellipsoid (CSE) along the major axis
    source:
    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{x^2 + \\frac{y^2}{q^2}}

    """
    param_names = ['A', 's', 'q', 'center_x', 'center_y']
    lower_limit_default = {'A': -1000, 's': 0, 'q': 0.001, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'A': 1000, 's': 10000, 'q': 0.99999, 'e2': 0.5, 'center_x': -100, 'center_y': -100}

    def function(self, x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: lensing potential
        """

        # potential calculation
        psi = np.sqrt(q**2*(s**2 + x**2) + y**2)
        Phi = (psi + s)**2 + (1-q**2) * x**2
        phi = q/(2*s) * np.log(Phi) - q/s * np.log((1+q) * s)
        return a * phi

    def derivatives(self, x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """

        psi = np.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        Phi = (psi + s) ** 2 + (1 - q ** 2) * x ** 2
        f_x = q * x * (psi + q**2*s) / (s * psi * Phi)
        f_y = q * y * (psi + s) / (s * psi * Phi)

        return a * f_x, a * f_y

    def hessian(self, x, y, a, s, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """

        # equations 21-23 in Oguri 2021
        psi = np.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        Phi = (psi + s) ** 2 + (1 - q ** 2) * x ** 2
        f_xx = q/(s * Phi) * (1 + q**2*s*(q**2 * s**2 + y**2)/psi**3 - 2*x**2*(psi + q**2*s)**2/(psi**2 * Phi))
        f_yy = q/(s * Phi) * (1 + q**2 * s * (s**2 + x**2)/psi**3 - 2*y**2*(psi + s)**2/(psi**2 * Phi))
        f_xy = - q * x*y / (s * Phi) * (q**2 * s / psi**3 + 2 * (psi + q**2*s) * (psi + s) / (psi**2 * Phi))

        return a * f_xx, a * f_xy, a * f_xy, a * f_yy


class CSEMajorAxisSet(LensProfileBase):
    """A set of CSE profiles along a joint center and axis."""

    def __init__(self):
        self.major_axis_model = CSEMajorAxis()
        super(CSEMajorAxisSet, self).__init__()

    def function(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: lensing potential
        """
        f_ = np.zeros_like(x)
        for a, s in zip(a_list, s_list):
            f_ += self.major_axis_model.function(x, y, a, s, q)
        return f_

    def derivatives(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        f_x, f_y = np.zeros_like(x), np.zeros_like(y)
        for a, s in zip(a_list, s_list):
            f_x_, f_y_ = self.major_axis_model.derivatives(x, y, a, s, q)
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        f_xx, f_xy, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for a, s in zip(a_list, s_list):
            f_xx_, f_xy_, _, f_yy_ = self.major_axis_model.hessian(x, y, a, s, q)
            f_xx += f_xx_
            f_xy += f_xy_
            f_yy += f_yy_
        return f_xx, f_xy, f_xy, f_yy


class CSEProductAvg(LensProfileBase):
    """Cored steep ellipsoid (CSE) evaluated at the product-averaged radius sqrt(ab),
    such that mass is not changed when increasing ellipticity.

    Same as CSEMajorAxis but evaluated at r=sqrt(q)*r_original

    Keeton and Kochanek (1998)
    Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with

    .. math::
        \\xi(x, y) = \\sqrt{qx^2 + \\frac{y^2}{q}}
    """
    param_names = ['A', 's', 'q', 'center_x', 'center_y']
    lower_limit_default = {'A': -1000, 's': 0, 'q': 0.001, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'A': 1000, 's': 10000, 'q': 0.99999, 'e2': 0.5, 'center_x': -100, 'center_y': -100}

    def __init__(self):
        super(CSEProductAvg, self).__init__()
        self.MA_class = CSEMajorAxis()

    @staticmethod
    def _convert2prodavg(x, y, a, s, q):
        """Converts coordinates and re-normalizes major-axis parameterization to instead
        be wrt.

        product-averaged
        """
        a = a / q
        x = x * np.sqrt(q)
        y = y * np.sqrt(q)
        return x, y, a, s, q

    def function(self, x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: lensing potential
        """
        x, y, a, s, q = self._convert2prodavg(x, y, a, s, q)
        return self.MA_class.function(x, y, a, s, q)

    def derivatives(self, x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        x, y, a, s, q = self._convert2prodavg(x, y, a, s, q)
        af_x, af_y = self.MA_class.derivatives(x, y, a, s, q)
        # extra sqrt(q) factor from taking derivative of transformed coordinate
        return np.sqrt(q) * af_x, np.sqrt(q) * af_y

    def hessian(self, x, y, a, s, q):
        """
        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        x, y, a, s, q = self._convert2prodavg(x, y, a, s, q)
        af_xx, af_xy, af_xy, af_yy = self.MA_class.hessian(x, y, a, s, q)
        # two sqrt(q) factors from taking derivatives of transformed coordinate
        return q * af_xx, q * af_xy, q * af_xy, q * af_yy


class CSEProductAvgSet(LensProfileBase):
    """A set of CSE profiles along a joint center and axis."""

    def __init__(self):
        self.major_axis_model = CSEProductAvg()
        super(CSEProductAvgSet, self).__init__()

    def function(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: lensing potential
        """
        f_ = np.zeros_like(x)
        for a, s in zip(a_list, s_list):
            f_ += self.major_axis_model.function(x, y, a, s, q)
        return f_

    def derivatives(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: deflection in x- and y-direction
        """
        f_x, f_y = np.zeros_like(x), np.zeros_like(y)
        for a, s in zip(a_list, s_list):
            f_x_, f_y_ = self.major_axis_model.derivatives(x, y, a, s, q)
            f_x += f_x_
            f_y += f_y_
        return f_x, f_y

    def hessian(self, x, y, a_list, s_list, q):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a_list: list of lensing strength
        :param s_list: list of core radius
        :param q: axis ratio
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        f_xx, f_xy, f_yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        for a, s in zip(a_list, s_list):
            f_xx_, f_xy_, _, f_yy_ = self.major_axis_model.hessian(x, y, a, s, q)
            f_xx += f_xx_
            f_xy += f_xy_
            f_yy += f_yy_
        return f_xx, f_xy, f_xy, f_yy
