__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Hessian']


class Hessian(LensProfileBase):
    """
    class for constant Hessian distortion (second order)
    The input is in the same convention as the LensModel.hessian() output.
    """
    param_names = ['f_xx', 'f_yy', 'f_xy', 'f_yx', 'ra_0', 'dec_0']
    lower_limit_default = {'f_xx': -100, 'f_yy': -100, 'f_xy': -100, 'f_yx': -100, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'f_xx': 100, 'f_yy': 100, 'f_xy': 100, 'f_yx': 100, 'ra_0': 100, 'dec_0': 100}

    def function(self, x, y, f_xx, f_yy, f_xy, f_yx, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param f_xx: dalpha_x/dx
        :param f_yy: dalpha_y/dy
        :param f_xy: dalpha_x/dy
        :param f_yx: dalpha_y/dx
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: lensing potential
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 1/2. * (f_xx * x_ * x_ + (f_xy + f_yx) * x_ * y_ + f_yy * y_ * y_)
        return f_

    def derivatives(self, x, y, f_xx, f_yy, f_xy, f_yx, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param f_xx: dalpha_x/dx
        :param f_yy: dalpha_y/dy
        :param f_xy: dalpha_x/dy
        :param f_yx: dalpha_y/dx
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: deflection angles
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = f_xx * x_ + f_xy * y_
        f_y = f_yx * x_ + f_yy * y_
        return f_x, f_y

    def hessian(self, x, y, f_xx, f_yy, f_xy, f_yx, ra_0=0, dec_0=0):
        """
        Hessian. Attention: If f_xy != f_yx then this function is not accurate!

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param f_xx: dalpha_x/dx
        :param f_yy: dalpha_y/dy
        :param f_xy: dalpha_x/dy
        :param f_yx: dalpha_y/dx
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: f_xx, f_yy, f_xy
        """
        return f_xx, f_xy, f_yx, f_yy
