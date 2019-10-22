from lenstronomy.GalKin import velocity_util as util


class PSF():
    """
    psf rendering module used in GalKin

    """
    def __init__(self, psf_type='GAUSSIAN', fwhm=0.7, moffat_beta=2.6):
        """

        :param psf_type: string, point spread functino type, current support for 'GAUSSIAN' and 'MOFFAT'
        :param fwhm: full width at half maximum seeing condition
        :param moffat_beta: float, beta parameter of Moffat profile
        """
        self._psf_type = psf_type
        self._fwhm = fwhm
        self._moffat_beta = moffat_beta

    def displace_psf(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: x', y' displaced by the two dimensional PSF distribution function
        """
        if self._psf_type == 'GAUSSIAN':
            return util.displace_PSF_gaussian(x, y, self._fwhm)
        elif self._psf_type == 'MOFFAT':
            return util.displace_PSF_moffat(x, y, self._fwhm, self._moffat_beta)
        else:
            raise ValueError('psf_type %s not supported for convolution!' % self._psf_type)
