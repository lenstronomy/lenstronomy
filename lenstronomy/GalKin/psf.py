from lenstronomy.GalKin import velocity_util as util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


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
            self._psf = PSF_GAUSSIAN(**kwargs_psf)
        elif psf_type == 'MOFFAT':
            self._psf = PSF_MOFFAT(**kwargs_psf)
        else:
            raise ValueError('psf_type %s not supported for convolution!' % psf_type)

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two dimensional PSF distribution function
        """
        return self._psf.displace_psf(x, y)


@export
class PSF_GAUSSIAN(object):
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


@export
class PSF_MOFFAT(object):
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
