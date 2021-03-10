from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import param_util
from lenstronomy.Util import derivative_util
import numpy as np

__all__ = ['ArcPerturbations']


class ArcPerturbations(LensProfileBase):
    """
    uses radial and tangential fourier modes within a specific range in both directions to perturb a lensing potential
    """
    def __init__(self):
        super(ArcPerturbations, self).__init__()
        self._2_pi = np.pi * 2

    def function(self, x, y, coeff, d_r, d_phi, center_x, center_y):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param coeff: float, amplitude of basis
        :param d_r: period of radial sinusoidal in units of angle
        :param d_phi: period of tangential sinusoidal in radian
        :param center_x: center of rotation for tangential basis
        :param center_y: center of rotation for tangential basis
        :return:
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        dphi_ = d_phi / self._2_pi
        phi_r = self._phi_r(r, d_r)
        phi_theta = self._phi_theta(phi, dphi_)
        return phi_r * phi_theta * coeff

    def derivatives(self, x, y, coeff, d_r, d_phi, center_x, center_y):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param coeff: float, amplitude of basis
        :param d_r: period of radial sinusoidal in units of angle
        :param d_phi: period of tangential sinusoidal in radian
        :param center_x: center of rotation for tangential basis
        :param center_y: center of rotation for tangential basis
        :return: f_x, f_y
        """
        r, theta = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        dphi_ = d_phi / self._2_pi
        d_phi_dr = self._d_phi_r(r, d_r) * self._phi_theta(theta, dphi_)
        d_phi_d_theta = self._d_phi_theta(theta, dphi_) * self._phi_r(r, d_r)
        x_ = x - center_x
        y_ = y - center_y
        dr_dx = derivative_util.d_r_dx(x_, y_)
        dr_dy = derivative_util.d_r_dy(x_, y_)
        d_theta_dx = derivative_util.d_phi_dx(x_, y_)
        d_theta_dy = derivative_util.d_phi_dy(x_, y_)
        f_x = d_phi_dr * dr_dx + d_phi_d_theta * d_theta_dx
        f_y = d_phi_dr * dr_dy + d_phi_d_theta * d_theta_dy
        return f_x * coeff, f_y * coeff

    def hessian(self, x, y, coeff, d_r, d_phi, center_x, center_y):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param coeff: float, amplitude of basis
        :param d_r: period of radial sinusoidal in units of angle
        :param d_phi: period of tangential sinusoidal in radian
        :param center_x: center of rotation for tangential basis
        :param center_y: center of rotation for tangential basis
        :return: f_xx, f_yy, f_xy
        """
        r, theta = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        dphi_ = d_phi / self._2_pi

        d_phi_dr = self._d_phi_r(r, d_r) * self._phi_theta(theta, dphi_)
        d_phi_dr2 = self._d_phi_r2(r, d_r) * self._phi_theta(theta, dphi_)
        d_phi_d_theta = self._d_phi_theta(theta, dphi_) * self._phi_r(r, d_r)
        d_phi_d_theta2 = self._d_phi_theta2(theta, dphi_) * self._phi_r(r, d_r)
        d_phi_dr_dtheta = self._d_phi_r(r, d_r) * self._d_phi_theta(theta, dphi_)
        x_ = x - center_x
        y_ = y - center_y
        dr_dx = derivative_util.d_r_dx(x_, y_)
        dr_dy = derivative_util.d_r_dy(x_, y_)
        d_theta_dx = derivative_util.d_phi_dx(x_, y_)
        d_theta_dy = derivative_util.d_phi_dy(x_, y_)
        dr_dxx = derivative_util.d_x_diffr_dx(x_, y_)
        dr_dxy = derivative_util.d_x_diffr_dy(x_, y_)
        dr_dyy = derivative_util.d_y_diffr_dy(x_, y_)
        d_theta_dxx = derivative_util.d_phi_dxx(x_, y_)
        d_theta_dyy = derivative_util.d_phi_dyy(x_, y_)
        d_theta_dxy = derivative_util.d_phi_dxy(x_, y_)

        f_xx = d_phi_dr2 * dr_dx**2 + d_phi_dr * dr_dxx + d_phi_d_theta2 * d_theta_dx**2 + d_phi_d_theta * d_theta_dxx + 2 * d_phi_dr_dtheta * dr_dx * d_theta_dx
        f_yy = d_phi_dr2 * dr_dy**2 + d_phi_dr * dr_dyy + d_phi_d_theta2 * d_theta_dy**2 + d_phi_d_theta * d_theta_dyy + 2 * d_phi_dr_dtheta * dr_dy * d_theta_dy
        f_xy = d_phi_dr2 * dr_dx * dr_dy + d_phi_dr * dr_dxy + d_phi_d_theta2 * d_theta_dx * d_theta_dy + d_phi_d_theta * d_theta_dxy + d_phi_dr_dtheta * dr_dx * d_theta_dy + d_phi_dr_dtheta * dr_dy * d_theta_dx
        return f_xx * coeff, f_xy * coeff, f_xy * coeff, f_yy * coeff

    @staticmethod
    def _phi_r(r, d_r):
        """

        :param r: numpy array, radius
        :param d_r: period of radial sinusoidal in units of angle
        :return: radial component of the potential
        """
        return np.cos(r/d_r)

    @staticmethod
    def _d_phi_r(r, d_r):
        """
        radial derivatives

        :param r: numpy array, radius
        :param d_r: period of radial sinusoidal in units of angle
        :return: radial component of the potential
        """
        return -np.sin(r / d_r) / d_r

    @staticmethod
    def _d_phi_r2(r, d_r):
        """
        radial second derivatives

        :param r: numpy array, radius
        :param d_r: period of radial sinusoidal in units of angle
        :return: radial component of the potential
        """
        return -np.cos(r / d_r) / d_r**2

    @staticmethod
    def _phi_theta(theta, d_theta):
        """

        :param theta: numpy array, orientation angle in 2pi convention
        :param d_theta: period of tangential sinusoidal in radian
        :return: tangential component of the potential
        """
        return np.cos(theta / d_theta)

    @staticmethod
    def _d_phi_theta(theta, d_theta):
        """
        tangential derivatives

        :param theta: numpy array, angle
        :param d_theta: period of tangential sinusoidal in radian
        :return: tangential component of the potential
        """
        return -np.sin(theta / d_theta) / d_theta

    @staticmethod
    def _d_phi_theta2(r, d_theta):
        """
        tangential derivatives

        :param r: numpy array, radius
        :param d_theta: period of tangential sinusoidal in radian
        :return: tangential component of the potential
        """
        return -np.cos(r / d_theta) / d_theta**2
