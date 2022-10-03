__author__ = 'ntessore'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.epl import EPL, EPLMajorAxis
from lenstronomy.LensModel.Profiles.multipole import Multipole

from scipy.special import hyp2f1

__all__ = ['EPL_boxydisky']

class EPL_boxydisky(LensProfileBase):
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
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y',\
                'a_m'] #, 'phi_m', 'center_x', 'center_y']
    # first line: for EPL
    # second line: for boxydisky; phi_m will be calculated, center_x and center_y will be shared.
    # m is set to be 4.
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100,\
                           'a_m': -0.1}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100,\
                           'a_m': +0.1}

    def __init__(self):
        self.epl_major_axis = EPLMajorAxis()
        self.multipole = Multipole()
        self.m = int(4)
        super(EPL_boxydisky, self).__init__()

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

    def _param_conv(self, theta_E, gamma, e1, e2):
        """
        convert parameters from :math:`R = r \sqrt{1 − e*cos(2*phi)}` to
        :math:`R = \sqrt{q^2 x^2 + y^2}`

        :param gamma: power law slope
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: critical radius b, slope t, axis ratio q, orientation angle phi_G
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E_conv = self._theta_E_q_convert(theta_E, q)
        b = theta_E_conv * np.sqrt((1 + q**2)/2)
        t = gamma - 1
        return b, t, q, phi_G

    def set_static(self, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
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

    def function(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
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
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2) # critical radius b, slope t, axis ratio q, orientation angle phi_G
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_epl = self.epl_major_axis.function(x__, y__, b, t, q)
        phi_m = phi_G
        # multipole should not be rotated beforehand...!!!
        f_multipole = self.multipole.function(x_, y_, self.m, a_m, phi_m)
        # rotate back
        return f_epl + f_multipole

    def function_epl_multipole_separately(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
        """
        Gives EPL and Multipole potentials separately.
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
        b, t, q, phi_G = self.param_conv(theta_E, gamma, e1, e2) # critical radius b, slope t, axis ratio q, orientation angle phi_G
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_epl = self.epl_major_axis.function(x__, y__, b, t, q)
        phi_m = phi_G
        f_multipole = self.multipole.function(x_, y_, self.m, a_m, phi_m)
        # rotate back
        return f_epl, f_multipole

    def derivatives(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
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
        f__x_epl, f__y_epl = self.epl_major_axis.derivatives(x__, y__, b, t, q)

        phi_m = phi_G
        f__x_multipole, f__y_multipole = self.multipole.derivatives(x_, y_, self.m, a_m, phi_m)
        # rotate back
        f__x = f__x_epl + f__x_multipole
        f__y = f__y_epl + f__y_multipole
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
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
        f__xx_epl, f__xy_epl, f__yx_epl, f__yy_epl = self.epl_major_axis.hessian(x__, y__, b, t, q)
        phi_m = phi_G
        f__xx_multipole, f__xy_multipole, f__yx_multipole, f__yy_multipole = self.multipole.hessian(x_, y_, self.m,
                                                                                                    a_m, phi_m)
        f__xx = f__xx_epl + f__xx_multipole
        f__xy = f__xy_epl = f__xy_multipole
        f__yx = f__yx_epl + f__yx_multipole
        f__yy = f__yy_epl + f__yy_multipole
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

    def _theta_E_q_convert(self, theta_E, q):
        """
        converts a spherical averaged Einstein radius to an elliptical (major axis) Einstein radius.
        This then follows the convention of the PEMD profile in lenstronomy.

        .. math::
            \\frac{\\theta_E}{\\theta_{E gravlens}}) = \\sqrt{(1+q^2) / (2 q)}

        :param theta_E: Einstein radius in lenstronomy conventions
        :param q: axis ratio minor/major
        :return: theta_E in convention of kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
        """
        theta_E_new = theta_E / (np.sqrt((1.+q**2) / (2. * q)))
        return theta_E_new
