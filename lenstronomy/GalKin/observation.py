from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin.psf import PSF
import lenstronomy.Util.util as util
from scipy.signal import convolve2d
import numpy as np

__all__ = ["GalkinObservation"]


class GalkinObservation(PSF, Aperture):
    """This class sets the base for the observational properties (aperture and seeing
    condition)"""

    def __init__(self, kwargs_aperture, kwargs_psf, backend):
        """

        :param kwargs_aperture: aperture parameters (see lenstronomy.GalKin.aperture)
        :param kwargs_psf: psf parameters (see lenstronomy.GalKin.psf)
        :param backend: either 'galkin' or 'jampy'
        """
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

    def convolve(self, data, supersampling_factor=1, num_pix=None):
        delta_pix_psf = self.delta_pix / supersampling_factor
        if num_pix is None:
            if self.psf_type == "PIXEL":
                num_pix = self._psf.kenrel_size
            else:
                num_pix = 6 * self.psf_fwhm / delta_pix_psf
                # make odd
                num_pix = int(np.ceil(num_pix)) // 2 * 2 + 1
        kernel = self.convolution_kernel(delta_pix_psf, num_pix)
        return convolve2d(data, kernel, mode="same")
