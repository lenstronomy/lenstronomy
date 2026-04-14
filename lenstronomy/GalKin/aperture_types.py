__author__ = "sibirrer"

import numpy as np
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


class ApertureBase:
    """General aperture class.

    This is intended to be inherited to define specific apertures. The aperture is
    defined by a set of coordinates (x_grid, y_grid) to be sampled, and then binned
    according to the bins matrix. The aperture can be supersampled and padded for PSF
    convolution.
    """

    def __init__(self, x_grid, y_grid, bins, delta_pix=0.1, padding_arcsec=0, angle=0):
        """

        :param x_grid: 2d array of x coordinates to compute the kinematics
        :param y_grid: 2d array of y coordinates to compute the kinematics
        :param bins: int array of shape (n_y, n_x) with the bin ids (0, 1, ...),
            and -1 for excluded pixels.
        :param delta_pix: spacing of the points to be sampled
        :param padding_arcsec: 0-padding for convolution in arcsec,
            this will be applied on all the edges of the aperture
        :param angle: position angle of the grid in radians
        """
        self._x_grid = np.asarray(x_grid)
        self._y_grid = np.asarray(y_grid)
        self._delta_pix = delta_pix
        self._padding_arcsec = padding_arcsec
        self._angle = angle
        self._bins = np.asarray(bins, dtype=int)

    def aperture_sample(self, supersampling_factor):
        """Returns a grid of points within the aperture, with supersampling and padding.

        :param supersampling_factor: supersampling factor for the grid
        :return: regular (x, y) meshgrid within the aperture to be sampled.
        """
        padding = self.padding_pix(supersampling_factor)
        x_grid_sup, y_grid_sup = make_supersampled_grid(
            self._x_grid, self._y_grid, supersampling_factor, padding, self._angle
        )
        return x_grid_sup, y_grid_sup

    def aperture_downsample(self, aperture_samples, supersampling_factor):
        """Downsamples the aperture to the desired bins.

        :param aperture_samples: map of values in a regular grid within the aperture
        :param supersampling_factor: supersampling factor for the grid
        :return: integrated values into num_segments
        """
        # remove padding
        padding = self.padding_pix(supersampling_factor)
        aperture_samples = _unpad_map(aperture_samples, padding)
        num_pix_y, num_pix_x = self._x_grid.shape
        # remove supersampling
        aperture_samples_unp = _undo_supersampling(
            aperture_samples, num_pix_x, num_pix_y, supersampling_factor
        )
        aperture_samples_bin = downsample_values_to_bins(
            aperture_samples_unp,
            self._bins,
        )
        return aperture_samples_bin

    def aperture_select(self, ra, dec):
        """Test if a point is within the aperture, and return the bin id if it is.

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, and bin
            id
        """
        bins = self.bins.flatten()
        in_grid, grid_loc = general_aperture_select(
            ra, dec, self._x_grid, self._y_grid, self._delta_pix
        )
        if in_grid:
            bin_id = bins[grid_loc]
            if bin_id > -1:
                return True, bin_id
        return False, None

    @property
    def bins(self):
        return self._bins

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return int(np.max(self._bins)) + 1

    @property
    def delta_pix(self):
        return self._delta_pix

    @property
    def padding_arcsec(self):
        return self._padding_arcsec

    def padding_pix(self, supersampling_factor):
        """

        :param supersampling_factor: supersampling factor for the grid
        :return: int, padding in pixels
        """
        delta_pix_sup = self.delta_pix / supersampling_factor
        padding_pix = int(self.padding_arcsec / delta_pix_sup)
        return padding_pix


@export
class Slit(ApertureBase):
    """Slit aperture description."""

    def __init__(
        self,
        length,
        width,
        center_ra=0,
        center_dec=0,
        angle=0,
        delta_pix=0.1,
        padding_arcsec=0,
    ):
        """

        :param length: length of slit
        :param width: width of slit
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit in radians,
            angle=0 corresponds length in RA direction
        :param delta_pix: size of the sub-pixels that samples the aperture for integration
        :param padding_arcsec: padding around the aperture for convolution
        """
        self._length = length
        self._width = width
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle
        x_grid, y_grid = make_slit_grid(
            delta_pix,
            self._length,
            self._width,
            self._center_ra,
            self._center_dec,
            self._angle,
        )
        bins = np.zeros_like(x_grid, dtype=int)
        super().__init__(x_grid, y_grid, bins, delta_pix, padding_arcsec, angle)

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


@export
def slit_select(ra, dec, length, width, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit in radians,
        angle=0 corresponds length in RA direction
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
def make_slit_grid(delta_pix, length, width, center_ra=0, center_dec=0, angle=0):
    """Creates a rectangular grid of points with an angle.

    :param delta_pix: size of the sub-pixels that samples the aperture for integration
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit in radians, angle=0 corresponds length in RA
        direction
    :return: bool, True if photon/ray is within the slit, False otherwise
    """
    slit_x = np.arange((-length + delta_pix) / 2, length / 2, delta_pix)
    slit_y = np.arange((-width + delta_pix) / 2, width / 2, delta_pix)
    grid_x, grid_y = np.meshgrid(slit_x, slit_y)
    # rotate
    grid_x, grid_y = _rotate(grid_x, grid_y, angle=angle)
    # shift
    grid_x = grid_x + center_ra
    grid_y = grid_y + center_dec
    return grid_x, grid_y


@export
class Frame(ApertureBase):
    """Rectangular box with a hole in the middle (also rectangular), effectively a
    frame."""

    def __init__(
        self,
        width_outer,
        width_inner,
        center_ra=0,
        center_dec=0,
        angle=0,
        delta_pix=0.1,
        padding_arcsec=0.0,
    ):
        """

        :param width_outer: width of box to the outer parts
        :param width_inner: width of inner removed box
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit in radians,
            angle=0 corresponds length in RA direction
        :param delta_pix: size of the sub-pixels that samples the aperture for integration
        :param padding_arcsec: padding around the aperture for convolution
        """
        self._width_outer = width_outer
        self._width_inner = width_inner
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

        x_grid, y_grid = make_slit_grid(
            delta_pix,
            self._width_outer,
            self._width_outer,
            self._center_ra,
            self._center_dec,
            self._angle,
        )
        bins = np.zeros_like(x_grid, dtype=int)
        mask_inner = (np.abs(x_grid - center_ra) < width_inner / 2) & (
            np.abs(y_grid - center_dec) < width_inner / 2
        )
        bins[mask_inner] = -1

        super().__init__(x_grid, y_grid, bins, delta_pix, padding_arcsec, angle)

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


@export
def frame_select(ra, dec, width_outer, width_inner, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param width_outer: width of box to the outer parts
    :param width_inner: width of inner removed box
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit in radians,
        angle=0 corresponds length in RA direction
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
class Shell(ApertureBase):
    """Shell aperture."""

    def __init__(
        self, r_in, r_out, center_ra=0, center_dec=0, delta_pix=0.1, padding_arcsec=0.0
    ):
        """

        :param r_in: innermost radius to be selected
        :param r_out: outermost radius to be selected
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        :param delta_pix: size of the sub-pixels that samples the aperture for integration
        :param padding_arcsec: padding around the aperture for convolution
        """
        self._r_in, self._r_out = r_in, r_out
        self._center_ra, self._center_dec = center_ra, center_dec
        x_grid, y_grid = make_slit_grid(
            delta_pix,
            self._r_out * 2,
            self._r_out * 2,
            self._center_ra,
            self._center_dec,
        )
        r_grid = np.sqrt(x_grid**2 + y_grid**2)
        bins = np.zeros_like(x_grid, dtype=int)
        bins[r_grid < self._r_in] = -1
        bins[r_grid > self._r_out] = -1

        super().__init__(x_grid, y_grid, bins, delta_pix, padding_arcsec)

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
class IFUGrid(ApertureBase):
    """Class for an Integral Field Unit spectrograph with rectangular grid where the
    kinematics are measured."""

    def __init__(self, x_grid, y_grid, padding_arcsec=0, angle=0):
        """

        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        :param padding_arcsec: padding of the IFU grid for convolution in arcsec
        :param angle: angle of the IFU grid in radians
        """
        x0, y0 = _rotate(x_grid[0, 0], y_grid[0, 0], -angle)
        x1 = _rotate(x_grid[0, 1], y_grid[0, 1], -angle)[0]
        y1 = _rotate(x_grid[1, 0], y_grid[1, 0], -angle)[1]
        delta_x = x1 - x0
        delta_y = y1 - y0
        if not np.isclose(np.abs(delta_x), np.abs(delta_y), rtol=1e-3):
            raise ValueError(
                "The IFU grid is irregular: |delta_x| != |delta_y|, "
                "check if there is a rotation angle!"
            )
        delta_pix = np.abs(delta_x)
        bins = np.arange(np.size(x_grid), dtype=int).reshape(x_grid.shape)
        super().__init__(x_grid, y_grid, bins, delta_pix, padding_arcsec, angle)

    def aperture_downsample(self, aperture_samples, supersampling_factor):
        """Downsample a high-resolution map to the IFU grid by averaging over the
        supersampling factor.

        :param aperture_samples: 2D array of high-resolution map to be downsampled
        :param supersampling_factor: supersampling factor
        :return: 2D array of downsampled map
        """
        num_pix_y, num_pix_x = self.num_segments
        padding = self.padding_pix(supersampling_factor)
        aperture_samples_unp = _unpad_map(aperture_samples, padding)
        return _undo_supersampling(
            aperture_samples_unp,
            num_pix_x,
            num_pix_y,
            supersampling_factor,
        )

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
class IFUShells(ApertureBase):
    """Class for an Integral Field Unit spectrograph with azimuthal shells where the
    kinematics are measured."""

    def __init__(
        self, r_bins, center_ra=0, center_dec=0, delta_pix=0.1, padding_arcsec=0
    ):
        """

        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
         It starts with the innermost edge to the outermost edge.
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        :param delta_pix: pixel scale of the IFU grid, only used if ifu_grid is None.
        :param padding_arcsec: padding of the IFU grid for convolution
        """
        self._r_bins = r_bins
        self._center_ra, self._center_dec = center_ra, center_dec
        r_max = np.max(r_bins)
        x_grid, y_grid = make_slit_grid(
            delta_pix,
            2 * r_max,
            2 * r_max,
            center_ra,
            center_dec,
        )
        r_grid = np.sqrt(x_grid**2 + y_grid**2)
        bins = np.ones_like(x_grid, dtype=int) * -1
        for i in range(len(r_bins) - 1):
            mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
            bins[mask] = i
        super().__init__(x_grid, y_grid, bins, delta_pix, padding_arcsec)

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
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int.
        """
        return len(self._r_bins) - 1


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


class IFUBinned(ApertureBase):
    """Class for an Integral Field Unit spectrograph, with a binned (e.g. Voronoi)
    rectangular grid.

    It has the same grid definition as IFUGrid, and a matrix of bin ids, indicating to
    which bin each pixel belongs.
    """

    def __init__(self, x_grid, y_grid, bins, padding_arcsec=0, angle=0):
        """

        :param x_grid: float array of shape (n_y, n_x) with the x coordinates of the grid
        :param y_grid: float array of shape (n_y, n_x) with the y coordinates of the grid
        :param bins: int array of shape (n_y, n_x) with the bin ids (0, 1, ...), and -1 for excluded pixels.
        :param padding_arcsec: padding of the IFU grid for convolution
        :param angle: angle of the IFU grid in radians
        """
        x0, y0 = _rotate(x_grid[0, 0], y_grid[0, 0], -angle)
        x1 = _rotate(x_grid[0, 1], y_grid[0, 1], -angle)[0]
        y1 = _rotate(x_grid[1, 0], y_grid[1, 0], -angle)[1]
        delta_x = x1 - x0
        delta_y = y1 - y0
        if not np.isclose(np.abs(delta_x), np.abs(delta_y), rtol=1e-3):
            raise ValueError(
                "The IFU grid is irregular: |delta_x| != |delta_y|, "
                "check if there is a rotation angle!"
            )
        delta_pix = np.abs(delta_x)
        super(IFUBinned, self).__init__(
            x_grid, y_grid, bins, delta_pix, padding_arcsec, angle
        )

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, and bin id
        """
        in_grid, grid_loc = grid_ifu_select(ra, dec, self._x_grid, self._y_grid)
        if in_grid:
            bin_id = self.bins[grid_loc]
            if bin_id > -1:
                return True, bin_id
        return False, None

    @property
    def x_grid(self):
        """X coordinates of the grid."""
        return self._x_grid

    @property
    def y_grid(self):
        """Y coordinates of the grid."""
        return self._y_grid


class GeneralAperture(ApertureBase):
    """General aperture that allows to sample in any shape, using 1d arrays of
    coordinates and bins.

    This is not compatible with supersampling or padding. It is meant to be used with
    the jampy backend and with a non-pixelated PSF.
    """

    def __init__(self, x_cords, y_cords, bins=None, delta_pix=0.1):
        """

        :param x_cords: 1d array of x coordinates to compute the kinematics
        :param y_cords: 1d array of y coordinates to compute the kinematics
        :param bins: array with the same shape as x_cords/y_cords
            defining the bin index for each coordinate
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        super(GeneralAperture, self).__init__(
            x_cords,
            y_cords,
            bins,
            delta_pix,
            padding_arcsec=0,
        )

    def aperture_sample(self, supersampling_factor):
        if supersampling_factor > 1:
            raise ValueError(
                "Supersampling factor cannot be greater than 1 for general aperture."
            )
        return self._x_grid, self._y_grid

    def aperture_downsample(self, aperture_samples, supersampling_factor):
        if supersampling_factor > 1:
            raise ValueError(
                "Supersampling factor cannot be greater than 1 for general aperture."
            )
        aperture_samples_bin = downsample_values_to_bins(
            aperture_samples,
            self._bins,
        )
        return aperture_samples_bin


def general_aperture_select(ra, dec, x_cords, y_cords, delta_pix=0.1):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param x_cords: x coordinates of pixels
    :param y_cords: y coordinates of pixels
    :param delta_pix: pixel size
    :return: bool, True if within a pixel, index of the pixel
    """
    num_cords = np.size(x_cords)
    for i in range(num_cords):
        x, y = x_cords.flatten()[i], y_cords.flatten()[i]
        x_down = x - delta_pix / 2
        x_up = x + delta_pix / 2

        y_down = y - delta_pix / 2
        y_up = y + delta_pix / 2

        if (x_down <= ra <= x_up) and (y_down <= dec <= y_up):
            return True, i
    return False, None


def make_supersampled_grid(
    x_grid,
    y_grid,
    supersampling_factor=1,
    padding=0,
    angle=0,
):
    """Creates a new grid, supersampled and with padding for PSF convolution.

    :param x_grid: x coordinates of the original grid
    :param y_grid: y coordinates of the original grid
    :param supersampling_factor: supersampling factor
    :param padding: padding in pixels around the supersampled grid
    :param angle: position angle in radians
    """
    if (supersampling_factor > 1) or (padding > 0):
        ny, nx = x_grid.shape
        # rotate to align with RA axis
        x_grid, y_grid = _rotate(x_grid, y_grid, angle=-angle)
        delta_x = x_grid[0, 1] - x_grid[0, 0]
        delta_y = y_grid[1, 0] - y_grid[0, 0]

        # New (supersampled) pixel size
        new_delta_x = delta_x / supersampling_factor
        new_delta_y = delta_y / supersampling_factor

        # the padding is in supersampled pixels
        pad_x = padding * new_delta_x
        pad_y = padding * new_delta_y

        # grid bounds (pixel-centered)
        x_start = x_grid[0, 0] - 0.5 * delta_x * (1 - 1 / supersampling_factor) - pad_x
        x_end = x_grid[0, -1] + 0.5 * delta_x * (1 - 1 / supersampling_factor) + pad_x
        y_start = y_grid[0, 0] - 0.5 * delta_y * (1 - 1 / supersampling_factor) - pad_y
        y_end = y_grid[-1, 0] + 0.5 * delta_y * (1 - 1 / supersampling_factor) + pad_y

        xs = np.linspace(x_start, x_end, nx * supersampling_factor + 2 * padding)
        ys = np.linspace(y_start, y_end, ny * supersampling_factor + 2 * padding)

        x_grid_supersampled, y_grid_supersampled = np.meshgrid(xs, ys)
        # rotate back to position angle
        x_grid_supersampled, y_grid_supersampled = _rotate(
            x_grid_supersampled, y_grid_supersampled, angle=angle
        )
        return x_grid_supersampled, y_grid_supersampled
    else:
        return x_grid, y_grid


def _rotate(x, y, angle):
    """Rotate coordinates in anti-clockwise direction.

    :param x: x coordinates
    :param y: y coordinates
    :param angle: angle to rotate
    :return: rotated x, rotated y
    """
    x_rot = np.cos(angle) * x - np.sin(angle) * y
    y_rot = np.sin(angle) * x + np.cos(angle) * y
    return x_rot, y_rot


def _unpad_map(padded_map, padding):
    """

    :param padded_map: 2d array with padding
    :param padding: number of padding pixels
    :return: 2d array with padding removed
    """
    if padding > 0:
        return padded_map[padding:-padding, padding:-padding]
    else:
        return padded_map


def _undo_supersampling(supresampled_map, num_pix_x, num_pix_y, supersampling_factor):
    if supersampling_factor > 1:
        return supresampled_map.reshape(
            num_pix_y, supersampling_factor, num_pix_x, supersampling_factor
        ).mean(axis=(1, 3))
    else:
        return supresampled_map


@export
def downsample_values_to_bins(values, bins):
    """

    :param values: data to be binned
    :param bins: bin ids in the same shape as values
    :return: 1d array with binned values
    """
    n_bins = int(np.max(bins)) + 1
    binned_values = np.zeros(n_bins)
    for n in range(n_bins):
        binned_values[n] = np.mean(values[bins == n])
    return binned_values
