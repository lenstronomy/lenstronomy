from lenstronomy.GalKin import velocity_util
from lenstronomy.Util import kernel_util


from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()

import numpy as np


@export
class PSF(object):
    """General class to handle the PSF in the GalKin module for rendering the
    displacement of photons/spectro."""

    def __init__(self, psf_type, **kwargs_psf):
        """

        :param psf_type: string, point spread function type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param kwargs_psf: keyword argument describing the relevant parameters of the PSF.
        """
        if psf_type == "GAUSSIAN":
            self._psf = PSFGaussian(**kwargs_psf)
        elif psf_type == "MOFFAT":
            self._psf = PSFMoffat(**kwargs_psf)
        else:
            raise ValueError("psf_type %s not supported for convolution!" % psf_type)

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two-dimensional PSF distribution function
        """
        return self._psf.displace_psf(x, y)

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """
        return self._psf.convolution_kernel(delta_pix, num_pix)


@export
class PSFGaussian(object):
    """Gaussian PSF."""

    def __init__(self, fwhm):
        """

        :param fwhm: full width at half maximum seeing condition
        """
        self._fwhm = fwhm

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two-dimensional PSF distribution function
        """
        return velocity_util.displace_PSF_gaussian(x, y, self._fwhm)

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """

        kernel = kernel_util.kernel_gaussian(num_pix, delta_pix, self._fwhm)
        return kernel

    def get_psf_kernel(self, x, y):
        """
        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        sigma = self._fwhm / 2 / np.sqrt(2 * np.log(2))
        return np.exp(-(x**2 + y**2) / 2 / sigma**2) / (2 * np.pi * sigma**2)

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        if hasattr(self, "_fwhm"):
            return self._fwhm
        else:
            return None


@export
class PSFMoffat(object):
    """Moffat PSF."""

    def __init__(self, fwhm, moffat_beta):
        """

        :param fwhm: full width at half maximum seeing condition
        :param moffat_beta: float, beta parameter of Moffat profile
        """
        self._fwhm = fwhm
        self._moffat_beta = moffat_beta

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two-dimensional PSF distribution function
        """
        return velocity_util.displace_PSF_moffat(x, y, self._fwhm, self._moffat_beta)

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """

        kernel = kernel_util.kernel_moffat(
            num_pix=num_pix,
            delta_pix=delta_pix,
            fwhm=self._fwhm,
            moffat_beta=self._moffat_beta,
        )
        return kernel
