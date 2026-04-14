from lenstronomy.GalKin import velocity_util
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util
from lenstronomy.Util.package_util import exporter
import lenstronomy.Util.multi_gauss_expansion as mge
from lenstronomy.LightModel.Profiles.moffat import Moffat
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import numpy as np

export, __all__ = exporter()


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
        elif psf_type == "MULTI_GAUSSIAN":
            self._psf = PSFMultiGaussian(**kwargs_psf)
        elif psf_type == "PIXEL":
            self._psf = PSFPixel(**kwargs_psf)
        else:
            raise ValueError("psf_type %s not supported for convolution!" % psf_type)
        self.psf_type = psf_type

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

    def convolution_kernel_grid(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        return self._psf.convolution_kernel_grid(x, y)

    @property
    def psf_fwhm(self):
        """PSF FWHM in arcsec."""
        return self._psf.fwhm

    @property
    def psf_multi_gauss_sigmas(self):
        """Sigmas of a multi gaussian expansion of the PSF used in jampy."""
        return self._psf.multi_gauss_sigmas

    @property
    def psf_multi_gauss_amplitudes(self):
        """Amplitudes of a multi gaussian expansion of the PSF used in jampy."""
        return self._psf.multi_gauss_amplitudes


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

    def convolution_kernel_grid(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        sigma = util.fwhm2sigma(self._fwhm)
        return kernel_util.kernel_gaussian_grid(x, y, sigma)

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def multi_gauss_sigmas(self):
        """Sigmas of a multi gaussian expansion of the PSF."""
        return util.fwhm2sigma(self._fwhm)

    @property
    def multi_gauss_amplitudes(self):
        """Amplitudes of a multi gaussian expansion of the PSF."""
        return 1.0


@export
class PSFMoffat(object):
    """Moffat PSF."""

    def __init__(self, fwhm, moffat_beta, n_gauss_approx=10):
        """

        :param fwhm: full width at half maximum seeing condition
        :param moffat_beta: float, beta parameter of Moffat profile
        :param n_gauss_approx: int, number of Gaussian components to use in a MGE for Jampy
        """
        self._fwhm = fwhm
        self._moffat_beta = moffat_beta
        self._n_gauss_approx = n_gauss_approx
        self._multi_gauss_amps, self._multi_gauss_sigmas = self._moffat_multi_gaussian()

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

    def convolution_kernel_grid(self, x, y):
        """

        :param x: x-coordinate of light ray
        :param y: y-coordinate of light ray
        :return: psf value at x and y grid positions
        """
        return kernel_util.kernel_moffat_grid(x, y, self._fwhm, self._moffat_beta)

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def multi_gauss_sigmas(self):
        """Sigmas of a multi gaussian expansion of the PSF."""
        return self._multi_gauss_sigmas

    @property
    def multi_gauss_amplitudes(self):
        """Amplitudes of a multi gaussian expansion of the PSF."""
        return self._multi_gauss_amps

    def _moffat_multi_gaussian(self):
        """Approximate Moffat as a multi-gaussian kernel."""
        r_array = np.logspace(-2, np.log10(10 * self._fwhm), num=100)
        alpha = velocity_util.moffat_fwhm_alpha(self._fwhm, self._moffat_beta)
        psf_array = Moffat().function(
            x=r_array, y=0, amp=1, alpha=alpha, beta=self._moffat_beta
        )
        amps, sigmas, _ = mge.mge_1d(r_array, psf_array, N=self._n_gauss_approx)
        amps = np.asarray(amps)
        sigmas = np.asarray(sigmas)
        amps = amps / amps.sum()
        return amps, sigmas


class PSFMultiGaussian(object):
    """Multi-Gaussian PSF."""

    def __init__(self, amplitudes, sigmas, fwhm=None):
        """

        :param amplitudes: amplitudes of the multi-gaussian components
        :param sigmas: sigmas of a multi-gaussian PSF in arcseconds
        :param fwhm: full width at half maximum seeing condition
        """
        self._amplitudes = amplitudes / np.sum(amplitudes)
        self._sigmas = sigmas
        self._gaussian = Gaussian()
        if fwhm is None:
            kernel = self.convolution_kernel(delta_pix=0.01, num_pix=201)
            r, p = _radial_profile_from_kernel(kernel, pixel_scale=0.01, n_bins=100)
            fwhm = _fwhm_from_radial_profile(r, p)
        self._fwhm = fwhm

    def convolution_kernel(self, delta_pix, num_pix=21):
        """Normalized convolution kernel.

        :param delta_pix: pixel scale of kernel
        :param num_pix: number of pixels per axis of the kernel
        :return: 2d numpy array of kernel
        """
        kernel = np.zeros((num_pix, num_pix))
        x_grid, y_grid = util.make_grid(num_pix, delta_pix)
        x_grid = x_grid.reshape(num_pix, num_pix)
        y_grid = y_grid.reshape(num_pix, num_pix)
        for amp, sigma in zip(self._amplitudes, self._sigmas):
            kernel += (
                self._gaussian.function(x_grid, y_grid, amp=amp, sigma=sigma)
                * delta_pix**2
            )
        kernel /= np.sum(kernel)
        return kernel

    def convolution_kernel_grid(self, x, y):
        kernel = np.zeros_like(x)
        for amp, sigma in zip(self._amplitudes, self._sigmas):
            kernel += self._gaussian.function(x, y, amp=amp, sigma=sigma)
        kernel /= np.sum(kernel)
        return kernel

    def displace_psf(self, x, y):
        raise NotImplementedError("displace_psf not implemented for Multi-Gaussian PSF")

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def multi_gauss_sigmas(self):
        """Sigmas of a multi gaussian expansion of the PSF."""
        return self._sigmas

    @property
    def multi_gauss_amplitudes(self):
        """Amplitudes of a multi gaussian expansion of the PSF."""
        return self._amplitudes


class PSFPixel(object):
    """Pixelated PSF model over a supersampled grid."""

    def __init__(self, kernel, delta_pix, supersampling_factor, fwhm=None):
        """

        :param kernel: 2D numpy array of supersampled kernel values
        :param delta_pix: pixel scale of kernel, accounting for supersampling
        :param supersampling_factor: supersampling factor
        :param fwhm: full width at half maximum seeing condition
        """
        # check that the kernel shape is odd and square
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("kernel must be square")
        if kernel.shape[0] % 2 == 0:
            raise ValueError("kernel must be odd-sized")
        self._kernel = np.asarray(kernel)
        self._kernel_size = kernel.shape[0]
        self._delta_pix = delta_pix
        self._supersampling_factor = supersampling_factor
        if fwhm is None:
            r, p = _radial_profile_from_kernel(
                kernel, pixel_scale=delta_pix, n_bins=100
            )
            fwhm = _fwhm_from_radial_profile(r, p)
        self._fwhm = fwhm

    def convolution_kernel(self, delta_pix, num_pix):
        if num_pix != self._kernel_size:
            raise ValueError("PSF grid does not match kernel shape")
        if not np.isclose(delta_pix, self._delta_pix, rtol=0.01):
            raise ValueError("PSF delta_pix does not match kernel pixel scale")
        return self._kernel

    def convolution_kernel_grid(self, x, y):
        if np.shape(x)[0] != self._kernel_size:
            raise ValueError("PSF grid does not match kernel shape")
        return self._kernel

    def displace_psf(self, x, y):
        raise NotImplementedError("displace_psf not implemented for Pixel PSF")

    @property
    def fwhm(self):
        """Retrieve FWHM of PSF if stored as a private variable."""
        return self._fwhm

    @property
    def multi_gauss_sigmas(self):
        """Sigmas of a multi gaussian expansion of the PSF."""
        return None

    @property
    def multi_gauss_amplitudes(self):
        """Amplitudes of a multi gaussian expansion of the PSF."""
        return None

    @property
    def supersampling_factor(self):
        """Retrieve supersampling factor if stored as a private variable."""
        return self._supersampling_factor

    @property
    def kenrel_size(self):
        """Retrieve kernel size if stored as a private variable."""
        return self._kernel_size


def _radial_profile_from_kernel(kernel, pixel_scale, n_bins=100):
    """

    kernel: 2D array, assumed centered PSF/kernel
    pixel_scale: physical size of one pixel
    n_bins: number of radial bins
    returns: r_centers, radial_profile
    """
    ny, nx = kernel.shape
    y0 = (ny - 1) / 2.0
    x0 = (nx - 1) / 2.0

    y, x = np.indices(kernel.shape)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) * pixel_scale

    r_max = r.max()
    bins = np.linspace(0.0, r_max, n_bins + 1)
    which = np.digitize(r.ravel(), bins) - 1

    prof = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    flat = kernel.ravel()
    for k in range(flat.size):
        b = which[k]
        if 0 <= b < n_bins:
            prof[b] += flat[k]
            counts[b] += 1

    valid = counts > 0
    prof[valid] /= counts[valid]

    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return r_centers[valid], prof[valid]


def _fwhm_from_radial_profile(r, p):
    """

    r: 1D array of radii in arcseconds, increasing from 0
    p: 1D array of profile values at those radii
    returns: approximate FWHM in arcseconds
    """
    p0 = p[0]
    half = 0.5 * p0

    idx = np.where(p <= half)[0]
    i = idx[0]

    # linear interpolation between the two surrounding samples
    r1, r2 = r[i - 1], r[i]
    p1, p2 = p[i - 1], p[i]
    r_half = r1 + (half - p1) * (r2 - r1) / (p2 - p1)

    return 2.0 * r_half
