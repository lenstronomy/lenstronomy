import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates

__all__ = ["PixelGrid"]


class PixelGrid(Coordinates):
    """Class that manages a specified pixel grid (rectangular at the moment) and its
    coordinates."""

    def __init__(
        self,
        nx,
        ny,
        transform_pix2angle,
        ra_at_xy_0,
        dec_at_xy_0,
    ):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        """
        Coordinates.__init__(self, transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_at_xy_0,
                             dec_at_xy_0=dec_at_xy_0)
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)
        # self.primary_beam = None  # this needs to be set to be compatible with ImageModel class requirements

    def update_pixel_grid(self, ra_shift=None, dec_shift=None, phi_rot=None):
        """
        updates the coordinate grid with shifts and rotations

        :param ra_shift: shift of RA coordinates in pixel grid
        :type ra_shift: float or None
        :param dec_shift: shift in DEC coordinates in pixel grid
        :type dec_shift: float or None
        :param phi_rot: rotation angle applied to coordinate grid around ra_at_xy_0, dec_at_xy_0 [radian]
        :type phi_rot: float or None
        :return: new Coordinate() class and pixel grid
        """
        self.update_coord_transform(ra_shift=ra_shift, dec_shift=dec_shift, phi_rot=phi_rot)
        self._x_grid, self._y_grid = self.coordinate_grid(self._nx, self._ny)

    @property
    def num_pixel(self):
        """

        :return: number of pixels in the data
        """
        return self._nx * self._ny

    @property
    def num_pixel_axes(self):
        """

        :return: number of pixels per axis, nx ny
        """
        return self._nx, self._ny

    @property
    def width(self):
        """

        :return: width of data frame
        """
        return self._nx * self.pixel_width, self._ny * self.pixel_width

    @property
    def center(self):
        """
        center RA, DEC of original coordinate frame (not including shift and rotations)

        :return: center_x, center_y of coordinate system
        """
        return np.mean(self._x_grid), np.mean(self._y_grid)

    @property
    def pixel_coordinates(self):
        """
        coordinates (2d) of the pixel grid

        :return: RA coords, DEC coords
        """
        return self._x_grid, self._y_grid
