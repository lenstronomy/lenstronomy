import scipy
import scipy.signal.signaltools as signaltools
import numpy as np

"""
fft convolution routines optimized for different scipy versions
"""


class FFT(object):
    """
    class for fast convolution computation with given image size and fixed kernel
    """
    def __init__(self, image, kernel):
        """

        :param kernel: convolution kernel (2d numpy array)
        :param shape: int list of shape of image to be convolved
        """
        import numpy.fft
        self._rfftn = np.fft.rfftn(kernel, np.shape(image))
        # numpy modules
        fftn = numpy.fft.fftn
        ifftn = numpy.fft.ifftn
        # scipy modules
        fftn = scipy.fftpack.fftn
        ifftn = scipy.fftpack.ifftn
        # FFTW numpy Monkey patch


        # FFTW scipy Monke patch



        # full FFTW
        import pyfftw
        import astropy.convolution
        a = pyfftw.empty_aligned((np.shape(image)), dtype='float32')
        b = pyfftw.empty_aligned((np.shape(kernel)), dtype='float32')
        fft = pyfftw.FFTW(a, b)
        ifft = numpy.fft.ifft
        self._kernel_fft = fft(kernel)

    def fft_convolve(self, image):
        """

        :param image: 2d array to match "shape"/rfftn kernel
        :return:
        """
        np.fft.fftn()

        #astropy.convolution.convolve_fft(array, kernel, boundary='fill', fill_value=0.0, nan_treatment='interpolate', normalize_kernel=True, normalization_zero_tol=1e-08, preserve_nan=False, mask=None, crop=True, return_fft=False, fft_pad=None, psf_pad=None, quiet=False, min_wt=0.0, allow_huge=False, fftn=<function fftn>, ifftn=<function ifftn>, complex_dtype=<class 'complex'>)

def fftconvolve(in1, in2, int2_fft, mode="same"):
    """

    :param in1:
    :param in2:
    :param int2_fft:
    :param mode:
    :return:
    """
    if scipy.__version__ == '0.14.0':
        return _fftconvolve_14(in1, in2, int2_fft, mode)
    else:
        return _fftconvolve_18(in1, in2, int2_fft, mode)


# scipy-0.18.0 compatible
def _fftconvolve_18(in1, in2, int2_fft, mode="same"):
    """
    scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
    """
    in1 = signaltools.asarray(in1)
    in2 = signaltools.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return signaltools.array([])

    s1 = signaltools.array(in1.shape)
    s2 = signaltools.array(in2.shape)

    shape = s1 + s2 - 1

    # Check that input sizes are compatible with 'valid' mode
    if signaltools._inputs_swap_needed(mode, s1, s2):
        # Convolution is commutative; order doesn't have any effect on output
        in1, s1, in2, s2 = in2, s2, in1, s1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.

    ret = np.fft.irfftn(np.fft.rfftn(in1, fshape) * int2_fft, fshape)[fslice].copy()
    #np.fft.rfftn(in2, fshape)

    if mode == "full":
        return ret
    elif mode == "same":
        return signaltools._centered(ret, s1)
    elif mode == "valid":
        return signaltools._centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")


# scipy-0.14.0 compatible
def _fftconvolve_14(in1, in2, int2_fft, mode="same"):
    """
    scipy routine scipy.signal.fftconvolve with kernel already fourier transformed
    """
    in1 = signaltools.asarray(in1)
    in2 = signaltools.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return signaltools.array([])

    s1 = signaltools.array(in1.shape)
    s2 = signaltools.array(in2.shape)

    shape = s1 + s2 - 1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [signaltools._next_regular(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.

    ret = signaltools.irfftn(signaltools.rfftn(in1, fshape) *
                    int2_fft, fshape)[fslice].copy()
    #np.fft.rfftn(in2, fshape)

    if mode == "full":
        return ret
    elif mode == "same":
        return signaltools._centered(ret, s1)
    elif mode == "valid":
        return signaltools._centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")


# scipy-0.18.0 compatible
def _fftn_18(image, kernel):
    """
    return the fourier transpose of the kernel in same modes as image
    :param image:
    :param kernel:
    :return:
    """
    in1 = signaltools.asarray(image)
    in2 = signaltools.asarray(kernel)

    s1 = signaltools.array(in1.shape)
    s2 = signaltools.array(in2.shape)

    shape = s1 + s2 - 1

    fshape = [signaltools.fftpack.helper.next_fast_len(int(d)) for d in shape]
    kernel_fft = np.fft.rfftn(in2, fshape)
    return kernel_fft


# scipy-0.14.0 compatible
def _fftn_14(image, kernel):
    """
    return the fourier transpose of the kernel in same modes as image
    :param image:
    :param kernel:
    :return:
    """
    in1 = signaltools.asarray(image)
    in2 = signaltools.asarray(kernel)

    s1 = signaltools.array(in1.shape)
    s2 = signaltools.array(in2.shape)

    shape = s1 + s2 - 1

    fshape = [signaltools._next_regular(int(d)) for d in shape]
    kernel_fft = signaltools.rfftn(in2, fshape)
    return kernel_fft


def fftn(image, kernel):
    """
    return the fourier transpose of the kernel in same modes as image
    :param image:
    :param kernel:
    :return:
    """
    if scipy.__version__ == '0.14.0':
        return _fftn_14(image, kernel)
    else:
        return _fftn_18(image, kernel)