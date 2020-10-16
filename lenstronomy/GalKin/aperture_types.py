__author__ = 'sibirrer'

import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class Slit(object):
    """
    Slit aperture description
    """

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
        return slit_select(ra, dec, self._length, self._width, self._center_ra, self._center_dec, self._angle), 0

    @property
    def num_segments(self):
        """
        number of segments with separate measurements of the velocity dispersion
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
    y = - np.sin(angle) * ra_ + np.cos(angle) * dec_

    if abs(x) < length / 2. and abs(y) < width / 2.:
        return True
    else:
        return False


@export
class Frame(object):
    """
    rectangular box with a hole in the middle (also rectangular), effectively a frame
    """

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
        return frame_select(ra, dec, self._width_outer, self._width_inner, self._center_ra, self._center_dec, self._angle), 0

    @property
    def num_segments(self):
        """
        number of segments with separate measurements of the velocity dispersion
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
    y = - np.sin(angle) * ra_ + np.cos(angle) * dec_
    if abs(x) < width_outer / 2. and abs(y) < width_outer / 2.:
        if abs(x) < width_inner / 2. and abs(y) < width_inner / 2.:
            return False
        else:
            return True
    return False


@export
class Shell(object):
    """
    Shell aperture
    """

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
        return shell_select(ra, dec, self._r_in, self._r_out, self._center_ra, self._center_dec), 0

    @property
    def num_segments(self):
        """
        number of segments with separate measurements of the velocity dispersion
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
    R = np.sqrt(x ** 2 + y ** 2)
    if (R >= r_in) and (R < r_out):
        return True
    else:
        return False


@export
class IFUShells(object):
    """
    class for an Integral Field Unit spectrograph with azimuthal shells where the kinematics are measured
    """
    def __init__(self, r_bins, center_ra=0, center_dec=0):
        """

        :param r_bins: array of radial bins to average the dispersion spectra in ascending order.
        It starts with the inner-most edge to the outermost edge.
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
        return shell_ifu_select(ra, dec, self._r_bins, self._center_ra, self._center_dec)

    @property
    def num_segments(self):
        """
        number of segments with separate measurements of the velocity dispersion
        :return: int
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
    R = np.sqrt(x ** 2 + y ** 2)
    for i in range(0, len(r_bin) - 1):
        if (R >= r_bin[i]) and (R < r_bin[i+1]):
            return True, i
    return False, None
