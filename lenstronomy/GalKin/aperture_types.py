__author__ = "sibirrer"

import numpy as np

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


class GeneralAperture(object):
    """General aperture class."""

    def __init__(self, x_cords, y_cords, bin_ids=None, delta_pix=0.1):
        """

        :param x_cords: x coordinates to compute the kinematics
        :param y_cords: y coordinates to compute the kinematics
        :param bin_ids: array with the same shape as x_cords/y_cords
            defining the bin index for each coordinate, -1 for excluded coordinates
        :param delta_pix: pixel scale of the coordinates, needed for PSF convolution
        """
        self._x_cords = x_cords
        self._y_cords = y_cords
        self._bin_ids = bin_ids
        self._delta_pix = delta_pix

    def aperture_sample(self):
        return self._x_cords, self._y_cords

    def aperture_downsample(self, high_res_map):
        return _downsample_values_to_bins(
            high_res_map.flatten(),
            self._bin_ids.flatten(),
        )

    def aperture_select(self, ra, dec):
        """
        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, and bin id
        """
        bin_ids = self._bin_ids.flatten()
        if bin_ids is None:
            bin_ids = np.arange(0, len(self._x_cords), 1)
        in_grid, grid_loc = general_aperture_select(
            ra, dec, self._x_cords, self._y_cords, self._delta_pix)
        if in_grid:
            bin_id = bin_ids[grid_loc]
            if bin_id > -1:
                return True, bin_id
        return False, None

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.

        :return: int
        """
        return int(np.max(self._bin_ids)) + 1

    @property
    def delta_pix(self):
        return self._delta_pix


def general_aperture_select(
    ra,
    dec,
    x_cords,
    y_cords,
    delta_pix=0.1
):
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

@export
class Slit(GeneralAperture):
    """Slit aperture description."""

    def __init__(self, length, width, center_ra=0, center_dec=0, angle=0, delta_pix=0.1):
        """

        :param length: length of slit
        :param width: width of slit
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        :param delta_pix: size of the sub-pixels that samples the aperture for integration
        """
        self._length = length
        self._width = width
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

        slit_grid_x, slit_grid_y = make_slit_grid(
            delta_pix, length, width, center_ra, center_dec, angle
        )
        super().__init__(
            slit_grid_x.flatten(), slit_grid_y.flatten(), delta_pix=delta_pix
        )

    def aperture_downsample(self, high_res_map):
        """
        Integrates the slit samples
        :param high_res_map: slit samples in a regular grid
        :return: a single integrated value
        """
        return np.mean(high_res_map)

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
def make_slit_grid(delta_pix, length, width, center_ra=0, center_dec=0, angle=0):
    """
    :param delta_pix: size of the sub-pixels that samples the aperture for integration
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool, True if photon/ray is within the slit, False otherwise
    """
    slit_x = np.arange((-length + delta_pix) / 2, length / 2, delta_pix)
    slit_y = np.arange((-width + delta_pix) / 2, width / 2, delta_pix)
    grid_x, grid_y = np.meshgrid(slit_x, slit_y)
    # rotate
    grid_x, grid_y = _rotate(grid_x, grid_y, angle=-angle)
    # shift
    grid_x = grid_x + center_ra
    grid_y = grid_y + center_dec
    return grid_x.flatten(), grid_y.flatten()


@export
class Frame(GeneralAperture):
    """Rectangular box with a hole in the middle (also rectangular), effectively a
    frame."""

    def __init__(self, width_outer, width_inner, center_ra=0, center_dec=0, angle=0, delta_pix=0.1):
        """

        :param width_outer: width of box to the outer parts
        :param width_inner: width of inner removed box
        :param center_ra: center of slit
        :param center_dec: center of slit
        :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
        :param delta_pix: size of the sub-pixels that samples the aperture for integration
        """
        self._width_outer = width_outer
        self._width_inner = width_inner
        self._center_ra, self._center_dec = center_ra, center_dec
        self._angle = angle

        x_grid, y_grid = make_frame_grid(
            delta_pix, width_outer, width_inner, center_ra, center_dec, angle
        )
        super().__init__(x_grid, y_grid, delta_pix=delta_pix)

    def aperture_downsample(self, high_res_map):
        """
        Integrates the frame samples
        :param high_res_map: frame samples in a regular grid
        :return: a single integrated value
        """
        return np.mean(high_res_map)

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
def make_frame_grid(delta_pix, width_inner, width_outer, center_ra=0, center_dec=0, angle=0):
    """Make a grid of coordinates within the frame aperture first create a grid for
    the outer box, then mask out the inner box
    :param delta_pix: size of the sub-pixels that samples the aperture for integration
    :param width_outer: width of box to the outer parts
    :param width_inner: width of inner removed box
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: x_grid, y_grid."""
    x_outer = np.arange(
        (-width_outer + delta_pix) / 2, width_outer / 2, delta_pix
    )
    y_outer = np.arange(
        (-width_outer + delta_pix) / 2, width_outer / 2, delta_pix
    )
    x_outer_grid, y_outer_grid = np.meshgrid(x_outer, y_outer)
    # rotate
    x_outer_grid, y_outer_grid = _rotate(
        x_outer_grid, y_outer_grid, angle=-angle
    )

    # create inner box mask
    mask_inner = (np.abs(x_outer_grid) < width_inner / 2) & (
            np.abs(y_outer_grid) < width_inner / 2
    )
    # apply mask
    x_grid = x_outer_grid[~mask_inner]
    y_grid = y_outer_grid[~mask_inner]
    # shift
    x_grid = x_grid + center_ra
    y_grid = y_grid + center_dec
    return x_grid, y_grid


@export
class Shell(GeneralAperture):
    """Shell aperture."""

    def __init__(self, r_in, r_out, center_ra=0, center_dec=0, delta_pix=0.1):
        """

        :param r_in: innermost radius to be selected
        :param r_out: outermost radius to be selected
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        """
        self._r_in, self._r_out = r_in, r_out
        self._center_ra, self._center_dec = center_ra, center_dec

        shell_x, shell_y = make_shell_grid(
            delta_pix, r_in, r_out, center_ra, center_dec
        )
        super().__init__(shell_x, shell_y, delta_pix=delta_pix)

    def aperture_downsample(self, high_res_map):
        """
        Integrates the shell samples
        :param high_res_map: shell samples in a regular grid
        :return: a single integrated value
        """
        return np.mean(high_res_map)

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


def make_shell_grid(delta_pix, r_in, r_out, center_ra=0, center_dec=0):
    """
    Make a grid of coordinates within the shell aperture,
    regularly sampled in polar coordinates
    :param delta_pix: size of the sub-pixels that samples the aperture for integration
    :param r_in: innermost radius to be selected
    :param r_out: outermost radius to be selected
    :param center_ra: center of the sphere
    :param center_dec: center of the sphere
    :return: x_grid, y_grid.
    """
    r_vals = np.arange(r_in, r_out, delta_pix)
    x_grid, y_grid = [], []
    for r in r_vals:
        x, y = _sample_circle_uniform(r, delta_pix)
        x_grid.append(x)
        y_grid.append(y)
    x_grid = np.concatenate(x_grid) + center_ra
    y_grid = np.concatenate(y_grid) + center_dec
    return x_grid, y_grid


@export
class IFUGrid(GeneralAperture):
    """Class for an Integral Field Unit spectrograph with rectangular grid where the
    kinematics are measured."""

    def __init__(self, x_grid, y_grid, supersampling_factor=1, padding_arcsec=0):
        """

        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        """
        self._x_grid = x_grid
        self._y_grid = y_grid

        delta_x = x_grid[0, 1] - x_grid[0, 0]
        delta_y = y_grid[1, 0] - y_grid[0, 0]
        if np.abs(delta_x) != np.abs(delta_y):
            raise ValueError("IFU grid pixels must be square!")
        delta_pix_sup = np.abs(delta_x) / supersampling_factor
        # padding in pixels
        self._padding = int(padding_arcsec / delta_pix_sup)
        self._supersampling_factor = supersampling_factor

        x_grid_supersampled, y_grid_supersampled = make_supersampled_ifu_grid(
            x_grid, y_grid, supersampling_factor, self._padding
        )
        super().__init__(
            x_grid_supersampled, y_grid_supersampled, delta_pix=delta_pix_sup
        )

    def aperture_downsample(self, high_res_map):
        """Downsample a high-resolution map to the IFU grid by averaging over the
        supersampling factor.

        :param high_res_map: 2D array of high-resolution map to be downsampled
        :return: 2D array of downsampled map
        """
        num_pix_y, num_pix_x = self.num_segments
        high_res_map = _unpad_map(high_res_map, self._padding)
        return high_res_map.reshape(
            num_pix_y, self._supersampling_factor, num_pix_x, self._supersampling_factor
        ).mean(axis=(1, 3))

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

    @property
    def supersampling_factor(self):
        """Supersampling factor of the IFU grid."""
        return self._supersampling_factor

    @property
    def padding(self):
        """Padding around the grid for convolution."""
        return self._padding


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


def make_supersampled_ifu_grid(x_grid, y_grid, supersampling_factor, padding):
    """Creates a new grid, supersampled and with padding for PSF convolution."""

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

    xs = np.arange(x_start, x_end * (1 + 1e-6), new_delta_x)
    ys = np.arange(y_start, y_end * (1 + 1e-6), new_delta_y)

    x_grid_supersampled, y_grid_supersampled = np.meshgrid(xs, ys)
    return x_grid_supersampled, y_grid_supersampled


@export
class IFUShells(IFUGrid):
    """Class for an Integral Field Unit spectrograph with azimuthal shells where the
    kinematics are measured."""

    def __init__(self, r_bins, center_ra=0, center_dec=0, ifu_grid_kwargs=None, delta_pix=None):
        """
        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
         It starts with the innermost edge to the outermost edge.
        :param center_ra: center of the sphere
        :param center_dec: center of the sphere
        :param ifu_grid_kwargs: kwargs to create the IFU grid, if None a default grid is created.
            the IFU grid is used to integrate the JAM equations, then binned into shells.
        :param delta_pix: pixel scale of the IFU grid, only used if ifu_grid is None.
        """
        self._r_bins = r_bins
        self._center_ra, self._center_dec = center_ra, center_dec
        if ifu_grid_kwargs is None:
            # make an IFU grid
            r_max = np.max(r_bins)
            if delta_pix is None:
                # default grid (same as in GalkinShells)
                delta_pix = r_max * 1.5 * 2 / 100
            ifu_x = ifu_y = np.arange(
                -r_max + delta_pix / 2,
                r_max,
                delta_pix,
            )
            ifu_x_grid, ifu_y_grid = np.meshgrid(ifu_x, ifu_y)
            ifu_grid_kwargs = {
                "x_grid": ifu_x_grid,
                "y_grid": ifu_y_grid,
                "supersampling_factor": 1,
                "padding_arcsec": 0,
            }
        super().__init__(**ifu_grid_kwargs)

    def aperture_downsample(self, high_res_map):
        """
        downsamples the IFU grid into shells
        :param high_res_map: supersampled IFU grid
        :return: measurements averaged into shells
        """
        downsampled_map = np.zeros(self.num_segments)
        x_grid, y_grid = self.aperture_sample()
        x_grid -= self._center_ra
        y_grid -= self._center_dec
        r_grid = np.sqrt(x_grid**2 + y_grid**2)
        r_bins = self._r_bins
        # iterate over bin edges and average masked values
        for i in range(self.num_segments):
            mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
            downsampled_map[i] = np.mean(high_res_map[mask])
        return downsampled_map

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


class IFUBinned(IFUGrid):
    """Class for an Integral Field Unit spectrograph, with a binned (e.g. Voronoi)
    rectangular grid.

    It has the same grid definition as IFUGrid, and a matrix of bin ids, indicating to
    which bin each pixel belongs.
    """

    def __init__(self, x_grid, y_grid, bins):
        """
        :param x_grid: float array of shape (n_y, n_x) with the x coordinates of the grid
        :param y_grid: float array of shape (n_y, n_x) with the y coordinates of the grid
        :param bins: int array of shape (n_y, n_x) with the bin ids (0, 1, ...), and -1 for excluded pixels.
        """
        super(IFUBinned, self).__init__(x_grid, y_grid)
        self._bins = bins.astype(int)

    def aperture_downsample(self, hires_map):
        """
        downsamples from a supersampled and padded IFU grid into bins
        :param hires_map: 2d array of values
        :return: 1d array of binned values
        """
        # remove padding from the grid
        hires_map = _unpad_map(hires_map, self.padding)
        # supersample bins to match the grid
        supersampled_bins = self.bins.repeat(self.supersampling_factor, axis=0).repeat(
            self.supersampling_factor, axis=1
        )
        # apply binning
        return _downsample_values_to_bins(hires_map, supersampled_bins)

    def aperture_select(self, ra, dec):
        """
        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, and bin id
        """
        in_grid, grid_loc = super(IFUBinned, self).aperture_select(ra, dec)
        if in_grid:
            bin_id = self.bins[grid_loc]
            if bin_id > -1:
                return True, bin_id
        return False, None

    @property
    def num_segments(self):
        """Number of segments with separate measurements of the velocity dispersion.
        This is the number of unique bin ids.

        :return: int.
        """
        unique_bins = np.unique(self._bins[self.bins > -1])
        return len(unique_bins)

    @property
    def bins(self):
        return self._bins


def _rotate(x, y, angle):
    """
    :param x: x coordinates
    :param y: y coordinates
    :param angle: angle to rotate
    :return: rotated x, rotated y
    """
    x_rot = np.cos(angle) * x + np.sin(angle) * y
    y_rot = -np.sin(angle) * x + np.cos(angle) * y
    return x_rot, y_rot


def _sample_circle_uniform(r_shell, step):
    """
    uniformly samples points in a circle
    :param r_shell: radius of the circle
    :param step: separation within points
    :return: x, y coordinates
    """
    n_points = 2 * np.pi * r_shell / step
    if np.ceil(n_points) > 1:
        angle = np.linspace(0, 2 * np.pi, int(np.ceil(n_points)))
    else:
        angle = np.array([0])
    return r_shell * np.cos(angle), r_shell * np.sin(angle)


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

def _downsample_values_to_bins(values, bins):
    """
    :param values: data to be binned
    :param bins: bin ids in the same shape as values
    :return: 1d array with binned values
    """
    n_bins = int(np.max(bins)) + 1
    vrms = np.zeros(n_bins)
    for n in range(n_bins):
        vrms[n] = np.mean(values[bins == n])
    return vrms
