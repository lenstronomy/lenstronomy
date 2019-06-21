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

    @property
    def transform_angle2pix(self):
        """

        :return: transformation matrix from angular to pixel coordinates
        """
        return self._Ma2pix

    @property
    def transform_pix2angle(self):
        """

        :return: transformation matrix from pixel to angular coordinates
        """
        return self._Mpix2a

    @property
    def xy_at_radec_0(self):
        """

        :return: pixel coordinate at angular (0,0) point
        """
        return self._x_at_radec_0, self._y_at_radec_0

    @property
    def radec_at_xy_0(self):
        """

        :return: RA, DEC coordinate at (0,0) pixel coordinate
        """
        return self._ra_at_xy_0, self._dec_at_xy_0

    def map_coord2pix(self, ra, dec):
        """
        maps the (ra,dec) coordinates of the system into the pixel coordinate of the image

        :param ra: relative RA coordinate as defined by the coordinate frame
        :param dec: relative DEC coordinate as defined by the coordinate frame
        :return: (x, y) pixel coordinates
        """

        return util.map_coord2pix(ra, dec, self._x_at_radec_0, self._y_at_radec_0, self._Ma2pix)

    def map_pix2coord(self, x, y):
        """
        maps the (x,y) pixel coordinates of the image into the system coordinates

        :param x: pixel coordinate (can be 1d numpy array), defined in the center of the pixel
        :param y: pixel coordinate (can be 1d numpy array), defined in the center of the pixel
        :return: relative (RA, DEC) coordinates of the system
        """
        return util.map_coord2pix(x, y, self._ra_at_xy_0, self._dec_at_xy_0, self._Mpix2a)

    @property
    def pixel_area(self):
        """
        angular area of a pixel in the image
        :return: area [arcsec^2]
        """
        return np.abs(linalg.det(self._Mpix2a))

    @property
    def pixel_width(self):
        """
        size of pixel
        :return: sqrt(pixel_area)
        """
        return np.sqrt(self.pixel_area)

    def coordinate_grid(self, nx, ny):
        """

        :param numPix: number of pixels per axis
        :return: 2d arrays with coordinates in RA/DEC with ra_coord[y-axis, x-axis]
        """
        ra_coords, dec_coords = util.grid_from_coordinate_transform(nx, ny, self._Mpix2a, self._ra_at_xy_0, self._dec_at_xy_0)
        ra_coords = util.array2image(ra_coords, nx, ny)  # new
        dec_coords = util.array2image(dec_coords, nx, ny)  # new
        return ra_coords, dec_coords

    def shift_coordinate_system(self, x_shift, y_shift, pixel_unit=False):
        """
        shifts the coordinate system
        :param x_shif: shift in x (or RA)
        :param y_shift: shift in y (or DEC)
        :param pixel_unit: bool, if True, units of pixels in input, otherwise RA/DEC
        :return: updated data class with change in coordinate system
        """
        self._shift_coordinates(x_shift, y_shift, pixel_unit)

    def _shift_coordinates(self, x_shift, y_shift, pixel_unit=False):
        """

        shifts the coordinate system
        :param x_shif: shift in x (or RA)
        :param y_shift: shift in y (or DEC)
        :param pixel_unit: bool, if True, units of pixels in input, otherwise RA/DEC
        :return: updated data class with change in coordinate system
        """
        if pixel_unit is True:
            ra_shift, dec_shift = self.map_pix2coord(x_shift, y_shift)
            ra_shift -= self._ra_at_xy_0
            dec_shift -= self._dec_at_xy_0
            print(ra_shift, dec_shift, 'test')
        else:
            ra_shift, dec_shift = x_shift, y_shift
        self._ra_at_xy_0 += ra_shift
        self._dec_at_xy_0 += dec_shift
        self._x_at_radec_0, self._y_at_radec_0 = util.map_coord2pix(-self._ra_at_xy_0, -self._dec_at_xy_0, 0, 0,
                                                                    self._Ma2pix)


class Coordinates1D(Coordinates):
    """
    coordinate grid described in 1-d arrays
    """
    def coordinate_grid(self, nx, ny):
        """

        :param numPix: number of pixels per axis
        :return: 2d arrays with coordinates in RA/DEC with ra_coord[y-axis, x-axis]
        """
        ra_coords, dec_coords = util.grid_from_coordinate_transform(nx, ny, self._Mpix2a, self._ra_at_xy_0, self._dec_at_xy_0)
        return ra_coords, dec_coords