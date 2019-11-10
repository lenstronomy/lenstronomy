__author__ = 'sibirrer'

from lenstronomy.GalKin.aperture_types import Shell, Slit



"""
class that defines the aperture of the measurement (e.g. slit, integral field spectroscopy regions etc)

Available aperture types:
-------------------------

'slit': length, width, center_ra, center_dec, angle
'shell': r_in, r_out, center_ra, center_dec

"""


def aperture_select(aperture_type, **kwargs_aperture):
    """

    initializes the observation condition and masks
    :param aperture_type: string
    :param kwargs_aperture: keyword arguments reflecting the aperture type chosen.
    We refer to the specific class instances for documentation.
    """
    if aperture_type == 'slit':
        return Slit(**kwargs_aperture)
    elif aperture_type == 'shell':
        return Shell(**kwargs_aperture)
    else:
        raise ValueError("aperture type %s not implemented!" % aperture_type)
