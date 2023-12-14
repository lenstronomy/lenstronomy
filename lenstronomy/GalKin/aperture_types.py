__author__ = "sibirrer"

import numpy as np

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
class Slit(object):
    """Slit aperture description."""

    def __init__(self, length, width, center_ra=0, center_dec=0, angle=0):
        """

        :param length: length of slit
        :param width: width of slit
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        """
        self._length = length
        self._width = width
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            slit_select(
                ra,
                dec,
                self._length,
                self._width,
                self._center_ra,
                self._center_dec,
                self._angle,
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


@export
def slit_select(ra, dec, length, width, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool, True if photon/ray is within the slit, False otherwise
    """
    ra_ = ra - center_ra
    dec_ = dec - center_dec
    x = np.cos(angle) * ra_ + np.sin(angle) * dec_
    y = -np.sin(angle) * ra_ + np.cos(angle) * dec_

    if abs(x) < length / 2.0 and abs(y) < width / 2.0:
        return True
    else:
        return False


@export
class Frame(object):
    """Rectangular box with a hole in the middle (also rectangular), effectively a
    frame."""

    def __init__(self, width_outer, width_inner, center_ra=0, center_dec=0, angle=0):
        """

        :param width_outer: width of box to the outer parts
        :param width_inner: width of inner removed box
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        """
        self._width_outer = width_outer
        self._width_inner = width_inner
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            frame_select(
                ra,
                dec,
                self._width_outer,
                self._width_inner,
                self._center_ra,
                self._center_dec,
                self._angle,
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


@export
def frame_select(ra, dec, width_outer, width_inner, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param width_outer: width of box to the outer parts
    :param width_inner: width of inner removed box
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool, True if photon/ray is within the box with a hole, False otherwise
    """
    ra_ = ra - center_ra
    dec_ = dec - center_dec
    x = np.cos(angle) * ra_ + np.sin(angle) * dec_
    y = -np.sin(angle) * ra_ + np.cos(angle) * dec_
    if abs(x) < width_outer / 2.0 and abs(y) < width_outer / 2.0:
        if abs(x) < width_inner / 2.0 and abs(y) < width_inner / 2.0:
            return False
        else:
            return True
    return False


@export
class Shell(object):
    """Shell aperture."""

    def __init__(self, r_in, r_out, center_ra=0, center_dec=0):
        """

        :param r_in: innermost radius to be selected
        :param r_out: outermost radius to be selected
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        """
        self._r_in, self._r_out = r_in, r_out
        self._center_ra, self._center_dec = center_ra, center_dec

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return (
            shell_select(
                ra, dec, self._r_in, self._r_out, self._center_ra, self._center_dec
            ),
            0,
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return 1


@export
def shell_select(ra, dec, r_in, r_out, center_ra=0, center_dec=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param r_in: innermost radius to be selected
    :param r_out: outermost radius to be selected
    :param center_ra: center of the sphere
    :param center_dec: center of the sphere
    :return: boolean, True if within the radial range, False otherwise
    """
    x = ra - center_ra
    y = dec - center_dec
    r = np.sqrt(x**2 + y**2)
    if (r >= r_in) and (r < r_out):
        return True
    else:
        return False


@export
class IFUShells(object):
    """Class for an Integral Field Unit spectrograph with azimuthal shells where the
    kinematics are measured."""

    def __init__(self, r_bins, center_ra=0, center_dec=0):
        """

        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
         It starts with the innermost edge to the outermost edge.
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        """
        self._r_bins = r_bins
        self._center_ra, self._center_dec = center_ra, center_dec

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, index of shell
        """
        return shell_ifu_select(
            ra, dec, self._r_bins, self._center_ra, self._center_dec
        )

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion
        :return: int."""
        return len(self._r_bins) - 1


@export
class IFUGrid(object):
    """Class for an Integral Field Unit spectrograph with rectangular grid where the
    kinematics are measured."""

    def __init__(self, x_grid, y_grid):
        """

        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        """
        self._x_grid = x_grid
        self._y_grid = y_grid

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, index of shell
        """
        return grid_ifu_select(ra, dec, self._x_grid, self._y_grid)

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return self._x_grid.shape[0], self._x_grid.shape[1]

    @property
    def x_grid(self):
        """X coordinates of the grid."""
        return self._x_grid

    @property
    def y_grid(self):
        """Y coordinates of the grid."""
        return self._y_grid


@export
def grid_ifu_select(ra, dec, x_grid, y_grid):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param x_grid: array of x_grid bins
    :param y_grid: array of y_grid bins
    :return: boolean, True if within the grid range, False otherwise
    """
    x_pixel_size = x_grid[0, 1] - x_grid[0, 0]
    y_pixel_size = y_grid[1, 0] - y_grid[0, 0]

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x_down = x_grid[i, j] - x_pixel_size / 2
            x_up = x_grid[i, j] + x_pixel_size / 2

            y_down = y_grid[i, j] - y_pixel_size / 2
            y_up = y_grid[i, j] + y_pixel_size / 2

            if (x_down <= ra <= x_up) and (y_down <= dec <= y_up):
                return True, (i, j)

    return False, None


@export
def shell_ifu_select(ra, dec, r_bin, center_ra=0, center_dec=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param r_bin: array of radial bins to average the dispersion spectra in ascending order.
     It starts with the inner-most edge to the outermost edge.
    :param center_ra: center of the sphere
    :param center_dec: center of the sphere
    :return: boolean, True if within the radial range, False otherwise
    """
    x = ra - center_ra
    y = dec - center_dec
    r = np.sqrt(x**2 + y**2)
    for i in range(0, len(r_bin) - 1):
        if (r >= r_bin[i]) and (r < r_bin[i + 1]):
            return True, i
    return False, None
