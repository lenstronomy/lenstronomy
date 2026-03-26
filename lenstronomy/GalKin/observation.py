from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin.psf import PSF
import lenstronomy.Util.util as util

__all__ = ["GalkinObservation"]


class GalkinObservation(PSF, Aperture):
    """This class sets the base for the observational properties (aperture and seeing
    condition)"""

    def __init__(self, kwargs_aperture, kwargs_psf, backend='galkin'):

        PSF.__init__(self, **kwargs_psf)

        kwargs_aperture = kwargs_aperture.copy()

        if ("delta_pix" not in kwargs_aperture) and ("x_grid" not in kwargs_aperture):
            # set the sampling of the aperture to FWHM / 3
            kwargs_aperture["delta_pix"] = min(self.psf_fwhm / 3, 0.1)

        if (self.psf_type == "PIXEL") or (backend == "galkin"):
            # pixelated PSF requires padding for convolution,
            # set it to 3 times the PSF sigma
            if "padding_arcsec" not in kwargs_aperture:
                # add a padding of 3 times the PSF sigma for convolution
                padding_arcsec = util.fwhm2sigma(self.psf_fwhm) * 3
                kwargs_aperture["padding_arcsec"] = padding_arcsec

        if self.psf_type == "PIXEL":
            self._default_supersampling_factor = kwargs_psf["supersampling_factor"]
        else:
            self._default_supersampling_factor = 1

        Aperture.__init__(self, **kwargs_aperture)
