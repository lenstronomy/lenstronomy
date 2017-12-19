import numpy.linalg as linalg
import numpy as np
import astrofunc.util as util


class Coordinates(object):
    """
    class to handle linear coordinate transformations of an image
    """
    def __init__(self, transform_pix2angle, ra_at_xy_0, dec_at_xy_0):
        self._Mpix2a = transform_pix2angle
        self._Ma2pix = linalg.inv(self._Mpix2a)
        self._ra_at_xy_0 = ra_at_xy_0
        self._dec_at_xy_0 = dec_at_xy_0
        self._x_at_radec_0, self._y_at_radec_0 = util.map_coord2pix(-self._ra_at_xy_0, -self._dec_at_xy_0, 0, 0, self._Ma2pix)

    def map_coord2pix(self, ra, dec):
        """

        :param ra: ra coordinates, relative
        :param dec: dec coordinates, relative
        :param x_0: pixel value in x-axis of ra,dec = 0,0
        :param y_0: pixel value in y-axis of ra,dec = 0,0
        :param M:
        :return:
        """

        return util.map_coord2pix(ra, dec, self._x_at_radec_0, self._y_at_radec_0, self._Ma2pix)

    def map_pix2coord(self, x_pos, y_pos):
        """

        :param x_pos:
        :param y_pos:
        :return:
        """
        return util.map_coord2pix(x_pos, y_pos, self._ra_at_xy_0, self._dec_at_xy_0, self._Mpix2a)

    @property
    def pixel_area(self):
        return np.abs(linalg.det(self._Mpix2a))

    @property
    def pixel_size(self):
        return np.sqrt(self.pixel_area)

