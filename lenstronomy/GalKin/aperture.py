__author__ = 'sibirrer'

from lenstronomy.GalKin.aperture_types import Shell, Slit


"""
class that defines the aperture of the measurement (e.g. slit, integral field spectroscopy regions etc)

Available aperture types:
-------------------------

'slit': length, width, center_ra, center_dec, angle
'shell': r_in, r_out, center_ra, center_dec

"""

class Aperture(object):
    """
    defines mask(s) of spectra, can handle IFU and single slit/box type data.
    """
    def __init__(self, aperture_type, **kwargs_aperture):
        """

        :param aperture_type: string
        :param kwargs_aperture: keyword arguments reflecting the aperture type chosen.
        We refer to the specific class instances for documentation.
        """
        if aperture_type == 'slit':
            self._aperture = Slit(**kwargs_aperture)
        elif aperture_type == 'shell':
            self._aperture = Shell(**kwargs_aperture)
        else:
            raise ValueError("aperture type %s not implemented!" % aperture_type)

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise
        """
        return self._aperture.aperture_select(ra, dec)
