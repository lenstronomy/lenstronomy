from lenstronomy.GalKin import velocity_util as util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()

import numpy as np

@export
class PSF(object):
    """
    general class to handle the PSF in the GalKin module for rendering the displacement of photons/spectro
    """
    def __init__(self, psf_type, **kwargs_psf):
        """

        :param psf_type: string, point spread function type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param kwargs_psf: keyword argument describing the relevant parameters of the PSF.
        """
        if psf_type == 'GAUSSIAN':
            self._psf = PSFGaussian(**kwargs_psf)
        elif psf_type == 'MOFFAT':
            self._psf = PSFMoffat(**kwargs_psf)
        else:
            raise ValueError('psf_type %s not supported for convolution!' % psf_type)

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two dimensional PSF distribution function
        """
        return self._psf.displace_psf(x, y)

    def get_psf_kernel(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: PSF function at x and y
        """
        return self._psf.get_psf_kernel(x, y)


@export
class PSFGaussian(object):
    """
    Gaussian PSF
    """
    def __init__(self, fwhm):
        """

        :param fwhm: full width at half maximum seeing condition
        """
        self._fwhm = fwhm

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two dimensional PSF distribution function
        """
        return util.displace_PSF_gaussian(x, y, self._fwhm)

    def get_psf_kernel(self, x, y):
        """
        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        sigma = self._fwhm / 2 / np.sqrt(2 * np.log(2))
        return np.exp( - (x**2 + y**2) / 2 / sigma**2) / (2 * np.pi * sigma**2)


@export
class PSFMoffat(object):
    """
    Moffat PSF
    """

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
        :return: x', y' displaced by the two dimensional PSF distribution function
        """
        return util.displace_PSF_moffat(x, y, self._fwhm, self._moffat_beta)

    def get_psf_kernel(self, x, y):
        """
        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        alpha = self._fwhm / 2 / np.sqrt(2**(1/self._moffat_beta) - 1)

        return (self._moffat_beta - 1) / (np.pi * alpha**2) / (1 + (x**2 +
                                        y**2) / alpha**2)**(self._moffat_beta)