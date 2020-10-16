from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin.psf import PSF

__all__ = ['GalkinObservation']


class GalkinObservation(PSF, Aperture):
    """
    this class sets the base for the observational properties (aperture and seeing condition)
    """
    def __init__(self, kwargs_aperture, kwargs_psf):
        Aperture.__init__(self, **kwargs_aperture)
        PSF.__init__(self, **kwargs_psf)
