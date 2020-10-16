__author__ = 'sibirrer'

from scipy import fftpack
import numpy as np
import lenstronomy.Util.analysis_util as analysis_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def correlation_2D(image):
    """
    #TODO document normalization output in units

    :param image: 2d image
    :return: 2d fourier transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)
    return np.abs(F2)


@export
def power_spectrum_2d(image):
    """

    :param image: 2d numpy array
    :return: 2d power spectrum in frequency units of the pixels
    """
    nx, ny = np.shape(image)
    corr2d = correlation_2D(image)
    return (corr2d / nx / ny) ** 2


@export
def power_spectrum_1d(image):
    """

    :param image: 2d numpy array
    :return: 1d radially averaged power spectrum of image in frequency units of pixels
    """
    psd2D = power_spectrum_2d(image)
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D, r = analysis_util.azimuthalAverage(psd2D)
    return psd1D, r
