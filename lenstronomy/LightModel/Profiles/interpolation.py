__author__ = 'sibirrer'

import scipy.interpolate
import numpy as np

import lenstronomy.Util.util as util

__all__ = ['Interpol']


class Interpol(object):
    """
    class which uses an interpolation of an image to compute the surface brightness

    parameters are
    'image': 2d numpy array of surface brightness (not integrated flux per pixel!)
    'center_x': coordinate of center of image in angular units (i.e. arc seconds)
    'center_y': coordinate of center of image in angular units (i.e. arc seconds)
    'phi_G': rotation of image relative to the rectangular ra-to-dec orientation
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
        :return: surface brightness from the model at coordinates (x, y)
        """
        x_, y_ = self.coord2image_pixel(x, y, center_x, center_y, phi_G, scale)
        return amp * self.image_interp(x_, y_, image)

    def image_interp(self, x, y, image):
        if not hasattr(self, '_image_interp'):
            # Setup the interpolator.
            # Note that 'x' and 'y' in this block only refer to first and second
            # image array axes. Outside this block it is more complicated.
            nx, ny = np.shape(image)
            image_bounds = np.zeros((nx + 2, ny + 2))
            nx0, ny0 = nx + 2, ny + 2
            image_bounds[1:-1, 1:-1] = image
            x_grid = np.linspace(start=-(nx0 - 1) / 2, stop=(nx0 - 1) / 2, num=nx0)
            y_grid = np.linspace(start=-(ny0 - 1) / 2, stop=(ny0 - 1) / 2, num=ny0)
            self._image_interp = scipy.interpolate.RectBivariateSpline(x_grid, y_grid, image_bounds, kx=1, ky=1, s=0)

        # y and x must be flipped in call to interpolator
        # (try reversing, the unit tests will fail)
        return self._image_interp(y, x, grid=False)

    def total_flux(self, image, scale, amp=1, center_x=0, center_y=0, phi_G=0):
        """
        sums up all the image surface brightness (image pixels defined in surface brightness at the coordinate of the pixel)
        times pixel area

        :param image: pixelized surface brightness
        :param scale: scale of the pixel in units of angle
        :param amp: linear scaling parameter of the surface brightness multiplicative with the initial image
        :param center_x: center of image in angular coordinates
        :param center_y: center of image in angular coordinates
        :param phi_G: rotation angle
        :return: total flux of the image
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

    def delete_cache(self):
        """delete the cached interpolated image"""
        if hasattr(self, '_image_interp'):
            del self._image_interp
