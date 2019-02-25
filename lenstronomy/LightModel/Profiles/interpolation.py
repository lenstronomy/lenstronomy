__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import lenstronomy.Util.util as util


class Interpol(object):
    """
    class which uses an interpolation of an image to compute the surface brightness

    parameters are
    'image': 2d numpy array of surface brightness
    'center_x': coordinate of center of image
    'center_y': coordinate of center of image
    'phi_G': rotation of image
    'scale': arcseconds per pixel of the image to be interpolated

    """
    param_names = ['image', 'amp', 'center_x', 'center_y', 'phi_G', 'scale']
    lower_limit_default = {'amp': 0, 'center_x': -1000, 'center_y': -1000, 'scale': 0.000000001, 'phi_G': -np.pi}
    upper_limit_default = {'amp': 1000000, 'center_x': 1000, 'center_y': 1000, 'scale': 10000000000, 'phi_G': np.pi}

    def __init__(self):
        pass

    def function(self, x, y, image=None, amp=1, center_x=0, center_y=0, phi_G=0, scale=1):
        """

        :param x: x-coordinate to evaluate surface brightness
        :param y: y-coordinate to evaluate surface brightness
        :param image: 2d numpy array (image) to be used to interpolate
        :param amp: amplitude of surface brightness scaling in respect of original image
        :param center_x: center of interpolated image
        :param center_y: center of interpolated image
        :param phi_G: rotation angle of simulated image in respect to input gird
        :param scale: pixel scale (in angular units) of the simulated image
        :return:
        """
        #self._check_interp(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy)
        n = len(np.atleast_1d(x))
        x_, y_ = self.coord2image_pixel(x, y, center_x, center_y, phi_G, scale)
        if n <= 1 and np.shape(x) == ():
            f_out = self.image_interp(x_, y_, image)
            return f_out[0][0]
        else:
            f_out = np.zeros(n)
            for i in range(n):
                f_out[i] = self.image_interp(x_[i], y_[i], image)
        return f_out * amp

    def image_interp(self, x, y, image):
        if not hasattr(self, '_image_interp'):
            nx, ny = np.shape(image)
            image_bounds = np.zeros((nx + 2, ny + 2))
            nx0, ny0 = nx + 2, ny + 2
            image_bounds[1:-1, 1:-1] = image
            x_grid = np.linspace(start=-(nx0 - 1) / 2, stop=(nx0 - 1) / 2, num=nx0)
            y_grid = np.linspace(start=-(ny0 - 1) / 2, stop=(ny0 - 1) / 2, num=ny0)
            self._image_interp = scipy.interpolate.RectBivariateSpline(y_grid, x_grid, image_bounds, kx=1, ky=1, s=0)
        return self._image_interp(y, x)

    def total_flux(self, image, scale, amp=1, center_x=0, center_y=0, phi_G=0):
        """

        :param image:
        :param scale:
        :param amp:
        :param center_x:
        :param center_y:
        :param phi_G:
        :return:
        """
        return np.sum(image) * scale**2 * amp

    @staticmethod
    def coord2image_pixel(ra, dec, center_x, center_y, phi_G, scale):
        """

        :param ra: angular coordinate
        :param dec: angular coordinate
        :param center_x: center of image in angular coordinates
        :param center_y: center of image in angular coordinates
        :param phi_G: rotation angle
        :param scale: pixel scale of image
        :return: pixel coordinates
        """
        ra_ = ra - center_x
        dec_ = dec - center_y
        x_ = ra_ / scale
        y_ = dec_ / scale
        x, y = util.rotate(x_, y_, phi_G)
        return x, y
