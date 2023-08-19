import numpy as np
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Data.angular_sensitivity import AngularSensitivity

__all__ = ["PixelGrid"]


class PixelGrid(Coordinates, AngularSensitivity):
    """Class that manages a specified pixel grid (rectangular at the moment) and its
    coordinates."""

    def __init__(
        self,
        nx,
        ny,
        transform_pix2angle,
        ra_at_xy_0,
        dec_at_xy_0,
        antenna_primary_beam=None,
    ):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        :param antenna_primary_beam: 2d numpy array with the same size of imaga_data;
         more descriptions of the primary beam can be found in the AngularSensitivity class
        """
        super(PixelGrid, self).__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)
        if antenna_primary_beam is not None:
            pbx, pby = np.shape(antenna_primary_beam)
            if (pbx, pby) != (nx, ny):
                raise ValueError(
                    "The primary beam should have the same size with the image data!"
                )
        AngularSensitivity.__init__(self, antenna_primary_beam)

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

        :return: center_x, center_y of coordinate system
        """
        return np.mean(self._x_grid), np.mean(self._y_grid)

    def shift_coordinate_system(self, x_shift, y_shift, pixel_unit=False):
        """Shifts the coordinate system :param x_shift: shift in x (or RA) :param
        y_shift: shift in y (or DEC) :param pixel_unit: bool, if True, units of pixels
        in input, otherwise RA/DEC :return: updated data class with change in coordinate
        system."""
        self._shift_coordinates(x_shift, y_shift, pixel_unit=pixel_unit)
        self._x_grid, self._y_grid = self.coordinate_grid(self._nx, self._ny)

    @property
    def pixel_coordinates(self):
        """

        :return: RA coords, DEC coords
        """
        return self._x_grid, self._y_grid
