__author__ = 'ntessore'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.spp import SPP
from scipy.special import hyp2f1


__all__ = ['EPL', 'EPLMajorAxis']


class EPL(LensProfileBase):
    """"
    Elliptical Power Law mass profile

    .. math::
        \\kappa(x, y) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta_{E}}{\\sqrt{q x^2 + y^2/q}} \\right)^{\\gamma-1}

    with :math:`\\theta_{E}` is the (circularized) Einstein radius,
    :math:`\\gamma` is the negative power-law slope of the 3D mass distributions,
    :math:`q` is the minor/major axis ratio,
    and :math:`x` and :math:`y` are defined in a coordinate sys- tem aligned with the major and minor axis of the lens.

    In terms of eccentricities, this profile is defined as

    .. math::
        \\kappa(r) = \\frac{3-\\gamma}{2} \\left(\\frac{\\theta'_{E}}{r \\sqrt{1 − e*\\cos(2*\\phi)}} \\right)^{\\gamma-1}

    with :math:`\\epsilon` is the ellipticity defined as

    .. math::
        \\epsilon = \\frac{1-q^2}{1+q^2}

    And an Einstein radius :math:`\\theta'_{\\rm E}` related to the definition used is

    .. math::
        \\left(\\frac{\\theta'_{\\rm E}}{\\theta_{\\rm E}}\\right)^{2} = \\frac{2q}{1+q^2}.

    The mathematical form of the calculation is presented by Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819.
    The current implementation is using hyperbolic functions. The paper presents an iterative calculation scheme,
    converging in few iterations to high precision and accuracy.

    A (faster) implementation of the same model using numba is accessible as 'EPL_NUMBA' with the iterative calculation
    scheme.
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.epl_major_axis = EPLMajorAxis()
        self.spp = SPP()
        super(EPL, self).__init__()

    def param_conv(self, theta_E, gamma, e1, e2):
        """
        converts parameters as defined in this class to the parameters used in the EPLMajorAxis() class

        :param theta_E: Einstein radius as defined in the profile class
        :param gamma: negative power-law slope
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus

        :return: b, t, q, phi_G
        """
        if self._static is True:
            return self._b_static, self._t_static, self._q_static, self._phi_G_static
        return self._param_conv(theta_E, gamma, e1, e2)

    @staticmethod
    def _param_conv(theta_E, gamma, e1, e2):
        """
        convert parameters from :math:`R = \\sqrt{q x^2 + y^2/q}` to
        :math:`R = \\sqrt{q^2 x^2 + y^2}`

        :param gamma: power law slope
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: critical radius b, slope t, axis ratio q, orientation angle phi_G
        """
        t = gamma - 1
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        b = theta_E * np.sqrt(q)
        return b, t, q, phi_G

    def set_static(self, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        self._b_static, self._t_static, self._q_static, self._phi_G_static = self._param_conv(theta_E, gamma, e1, e2)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, '_b_static'):
            del self._b_static
        if hasattr(self, '_t_static'):
            del self._t_static
        if hasattr(self, '_phi_G_static'):
            del self._phi_G_static
        if hasattr(self, '_q_static'):
            del self._q_static

    def function(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.epl_major_axis.function(x__, y__, b, t, q)
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.epl_major_axis.derivatives(x__, y__, b, t, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """

        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__xy, f__yx, f__yy = self.epl_major_axis.hessian(x__, y__, b, t, q)
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    def mass_3d_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """
        computes the spherical power-law mass enclosed (with SPP routine)
        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.mass_3d_lens(r, theta_E, gamma)

    def density_lens(self, r, theta_E, gamma, e1=None, e2=None):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius within the mass is computed
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param e1: eccentricity component (not used)
        :param e2: eccentricity component (not used)
        :return: mass enclosed a 3D radius r
        """
        return self.spp.density_lens(r, theta_E, gamma)


class EPLMajorAxis(LensProfileBase):
    """
    This class contains the function and the derivatives of the
    elliptical power law.

    .. math::
        \\kappa = (2-t)/2 * \\left[\\frac{b}{\\sqrt{q^2 x^2 + y^2}}\\right]^t

    where with :math:`t = \\gamma - 1` (from EPL class) being the projected power-law slope of the convergence profile,
    critical radius b, axis ratio q.

    Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819
    """
    param_names = ['b', 't', 'q', 'center_x', 'center_y']

    def __init__(self):

        super(EPLMajorAxis, self).__init__()

    def function(self, x, y, b, t, q):
        """
        returns the lensing potential

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: lensing potential
        """
        # deflection from method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # deflection potential, eq. (15)
        psi = (x*alpha_x + y*alpha_y)/(2 - t)

        return psi

    def derivatives(self, x, y, b, t, q):
        """
        returns the deflection angles

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: f_x, f_y
        """
        # elliptical radius, eq. (5)
        Z = np.empty(np.shape(x), dtype=complex)
        Z.real = q*x
        Z.imag = y
        R = np.abs(Z)
        R = np.maximum(R, 0.000000001)

        # angular dependency with extra factor of R, eq. (23)
        R_omega = Z*hyp2f1(1, t/2, 2-t/2, -(1-q)/(1+q)*(Z/Z.conj()))

        # deflection, eq. (22)
        alpha = 2/(1+q)*(b/R)**t*R_omega

        # return real and imaginary part
        alpha_real = np.nan_to_num(alpha.real, posinf=10**10, neginf=-10**10)
        alpha_imag = np.nan_to_num(alpha.imag, posinf=10**10, neginf=-10**10)

        return alpha_real, alpha_imag

    def hessian(self, x, y, b, t, q):
        """
        Hessian matrix of the lensing potential

        :param x: x-coordinate in image plane relative to center (major axis)
        :param y: y-coordinate in image plane relative to center (minor axis)
        :param b: critical radius
        :param t: projected power-law slope
        :param q: axis ratio
        :return: f_xx, f_yy, f_xy
        """
        R = np.hypot(q*x, y)
        R = np.maximum(R, 0.00000001)
        r = np.hypot(x, y)

        cos, sin = x/r, y/r
        cos2, sin2 = cos*cos*2 - 1, sin*cos*2

        # convergence, eq. (2)
        kappa = (2 - t)/2*(b/R)**t
        kappa = np.nan_to_num(kappa, posinf=10**10, neginf=-10**10)

        # deflection via method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # shear, eq. (17), corrected version from arXiv/corrigendum
        gamma_1 = (1-t)*(alpha_x*cos - alpha_y*sin)/r - kappa*cos2
        gamma_2 = (1-t)*(alpha_y*cos + alpha_x*sin)/r - kappa*sin2
        gamma_1 = np.nan_to_num(gamma_1, posinf=10**10, neginf=-10**10)
        gamma_2 = np.nan_to_num(gamma_2, posinf=10**10, neginf=-10**10)

        # second derivatives from convergence and shear
        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2

        return f_xx, f_xy, f_xy, f_yy
