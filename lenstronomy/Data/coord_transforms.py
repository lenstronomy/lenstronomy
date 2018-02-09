import numpy.linalg as linalg
import numpy as np
import lenstronomy.Util.util as util


class Coordinates(object):
    """
    class to handle linear coordinate transformations of a square pixel image
    """
    def __init__(self, transform_pix2angle, ra_at_xy_0, dec_at_xy_0):
        """
        initialize the coordinate-to-pixel transform and their inverse
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        """
        self._Mpix2a = transform_pix2angle
        self._Ma2pix = linalg.inv(self._Mpix2a)
        self._ra_at_xy_0 = ra_at_xy_0
        self._dec_at_xy_0 = dec_at_xy_0
        self._x_at_radec_0, self._y_at_radec_0 = util.map_coord2pix(-self._ra_at_xy_0, -self._dec_at_xy_0, 0, 0, self._Ma2pix)

    def map_coord2pix(self, ra, dec):
        """

        :param ra: ra coordinates, relative
        :param dec: dec coordinates, relative
        :return: (x, y) pixel position of coordinate (ra, dec)
        """

        return util.map_coord2pix(ra, dec, self._x_at_radec_0, self._y_at_radec_0, self._Ma2pix)

    def map_pix2coord(self, x_pos, y_pos):
        """

        :param x_pos: pixel position
        :param y_pos: pixel position
        :return: (ra, dec) coordinates of pixel position (x_pos, y_pos)
        """
        return util.map_coord2pix(x_pos, y_pos, self._ra_at_xy_0, self._dec_at_xy_0, self._Mpix2a)

    @property
    def pixel_area(self):
        """
        angular area of a pixel in the image
        :return: area [arcsec^2]
        """
        return np.abs(linalg.det(self._Mpix2a))

    @property
    def pixel_size(self):
        """
        size of pixel
        :return: sqrt(pixel_area)
        """
        return np.sqrt(self.pixel_area)

    def coordinate_grid(self, numPix):
        """

        :param numPix:
        :return:
        """
        ra_coords, dec_coords = util.grid_from_coordinate_transform(numPix, self._Mpix2a, self._ra_at_xy_0, self._dec_at_xy_0)
        return ra_coords, dec_coords