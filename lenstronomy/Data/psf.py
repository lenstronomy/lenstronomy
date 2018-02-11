import scipy.ndimage as ndimage
import scipy.signal as signal
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util


class PSF(object):
    """

    """
    def __init__(self, kwargs_psf):
        self._kwargs_psf = kwargs_psf
        self.psf_type = kwargs_psf.get('psf_type', 'NONE')
        if self.psf_type == 'GAUSSIAN':
            #self.psf_fwhm = kwargs_psf['psf_fwhm']
            #self.psf_trunc = kwargs_psf['psf_trunc']
            pass

    @property
    def kernel_point_source(self):
        if 'kernel_point_source' in self._kwargs_psf:
            return self._kwargs_psf['kernel_point_source']
        else:
            raise ValueError("kwargs_psf has no 'kernel_point_source' attribute!")

    @property
    def psf_error_map(self):
        if 'error_map' in self._kwargs_psf:
            return self._kwargs_psf['error_map']
        else:
            raise ValueError("kwargs_psf has no 'error_map' attribute!")

    def psf_convolution(self, grid, grid_scale, psf_subgrid=False, subgrid_res=1):
        """
        convolves a given pixel grid with a PSF
        """
        kwargs = self._kwargs_psf
        psf_type = self.psf_type
        if psf_type == 'NONE':
            return grid
        elif psf_type == 'GAUSSIAN':
            sigma = kwargs['sigma']/grid_scale
            if 'sigma_truncate' in kwargs:
                sigma_truncate = kwargs['sigma_truncate']
            else:
                sigma_truncate = 5.
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=sigma_truncate*sigma)
            return img_conv
        elif psf_type == 'PIXEL':
            if psf_subgrid:
                kernel = self._subgrid_kernel(subgrid_res)
            else:
                kernel = kwargs['kernel_pixel']
            img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)

    def _subgrid_kernel(self, subgrid_res):
        """

        :return:
        """
        kwargs = self._kwargs_psf
        if not hasattr(self, '_subgrid_kernel_out'):
            kernel = kernel_util.subgrid_kernel(kwargs['kernel_point_source'], subgrid_res, odd=True)
            n = len(kwargs['kernel_pixel'])
            n_new = n * subgrid_res
            if n_new % 2 == 0:
                n_new -= 1
            self._subgrid_kernel_out = kernel_util.cut_psf(kernel, psf_size=n_new)
        return self._subgrid_kernel_out

    def psf_fwhm(self, kwargs, deltaPix):
        """

        :param kwargs_psf:
        :return: psf fwhm in units of arcsec
        """
        psf_type = kwargs.get('psf_type', 'NONE')
        if psf_type == 'NONE':
            fwhm = 0
        elif psf_type == 'GAUSSIAN':
            sigma = kwargs['sigma']
            fwhm = util.sigma2fwhm(sigma)
        elif psf_type == 'PIXEL':
            kernel = kwargs['kernel_point_source']
            fwhm = kernel_util.fwhm_kernel(kernel) * deltaPix
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)
        return fwhm