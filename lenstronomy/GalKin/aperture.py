__author__ = "sibirrer"

from lenstronomy.GalKin.aperture_types import (
    Shell,
    Slit,
    IFUShells,
    Frame,
    IFUGrid,
    IFUBinned,
    downsample_values_to_bins,
)

__all__ = ["Aperture"]
"""Class that defines the aperture of the measurement (e.g. slit, integral field
spectroscopy regions etc).

Available aperture types:
-------------------------

'slit': length, width, center_ra, center_dec, angle
'shell': r_in, r_out, center_ra, center_dec
"""


class Aperture(object):
    """Defines mask(s) of spectra, can handle IFU and single slit/box type data."""

    def __init__(self, aperture_type, **kwargs_aperture):
        """

        :param aperture_type: string
        :param kwargs_aperture: keyword arguments reflecting the aperture type chosen.
         We refer to the specific class instances for documentation.
        """
        if aperture_type == "slit":
            self._aperture = Slit(**kwargs_aperture)
        elif aperture_type == "shell":
            self._aperture = Shell(**kwargs_aperture)
        elif aperture_type == "IFU_shells":
            self._aperture = IFUShells(**kwargs_aperture)
        elif aperture_type == "frame":
            self._aperture = Frame(**kwargs_aperture)
        elif aperture_type == "IFU_grid":
            self._aperture = IFUGrid(**kwargs_aperture)
        elif aperture_type == "IFU_binned":
            self._aperture = IFUBinned(**kwargs_aperture)
        else:
            raise ValueError(
                "aperture type %s not implemented! Available are 'slit', 'shell', 'IFU_shells'. "
                % aperture_type
            )
        self.aperture_type = aperture_type

    def aperture_select(self, ra, dec):
        """

        :param ra: angular coordinate of photon/ray
        :param dec: angular coordinate of photon/ray
        :return: bool, True if photon/ray is within the slit, False otherwise, int of the segment of the IFU
        """
        return self._aperture.aperture_select(ra, dec)

    def aperture_sample(self, supersampling_factor):
        """

        :return: regular (x, y) grid within the aperture to be sampled
        """
        return self._aperture.aperture_sample(supersampling_factor)

    def aperture_downsample(self, aperture_samples, supersampling_factor):
        """

        :param aperture_samples: regular grid of values within the aperture to be integrated
        :param supersampling_factor: supersampling factor
        :return: averaged values within the aperture into num_segments
        """
        return self._aperture.aperture_downsample(
            aperture_samples, supersampling_factor
        )

    @property
    def num_segments(self):
        return self._aperture.num_segments

    @property
    def delta_pix(self):
        return self._aperture.delta_pix
