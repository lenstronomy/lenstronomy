import numpy as np


class Aperture(object):
    """
    class that defines the aperture of the measurement (e.g. slit, integral field spectroscopy regions etc)

    Available aperture types:
    -------------------------

    'slit': length, width, center_ra, center_dec, angle
    'shell': r_in, r_out, center_ra, center_dec

    """
    def __init__(self, aperture_type='slit', psf_fwhm=0.7):
        """
        initializes the observation condition and masks
        :param aperture_type: string
        :param psf_fwhm: float
        """
        self._aperture_type = aperture_type
        self._fwhm = psf_fwhm

    def aperture_select(self, ra, dec, kwargs_aperture):
        """
        returns a bool list if the coordinate is within the aperture (list)
        :param ra:
        :param dec:
        :return:
        """
        if self._aperture_type == 'shell':
            bool_list = shell_select(ra, dec, **kwargs_aperture)
        elif self._aperture_type == 'slit':
            bool_list = slit_select(ra, dec, **kwargs_aperture)
        else:
            raise ValueError("aperture type %s not implemented!" % self._aperture_type)
        return bool_list


def slit_select(ra, dec, length, width, center_ra=0, center_dec=0, angle=0):
    """

    :param ra: angular coordinate of photon/ray
    :param dec: angular coordinate of photon/ray
    :param length: length of slit
    :param width: width of slit
    :param center_ra: center of slit
    :param center_dec: center of slit
    :param angle: orientation angle of slit, angle=0 corresponds length in RA direction
    :return: bool
    """
    ra_ = ra - center_ra
    dec_ = dec - center_dec
    x = np.cos(angle)*ra_ + np.sin(angle)*dec_
    y = - np.sin(angle)*ra_ + np.cos(angle)*dec_

    if abs(x) < length/2. and abs(y) < width/2.:
        return True
    else:
        return False


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
    R = np.sqrt(x**2 + y**2)
    if (R >= r_in) and (R < r_out):
        return True
    else:
        return False
