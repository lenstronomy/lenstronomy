

class AdaptivNumerics(object):
    """
    this class manages and computes a surface brightness convolved image in an adaptive approach.
    The strategie applied are:
    1.1 surface brightness computation only where significant flux is expected
    1.2 super sampled surface brightness only in regimes of high spacial variability in the surface brightness and at
    high contrast
    2.1 convolution only applied where flux is present (avoid convolving a lot of zeros)
    2.2 simplified Multi-Gaussian convolution in regimes of low contrast
    2.3 (super-) sampled PSF convolution only at high contrast of highly variable sources
    """



class SubgridConvolution(object):
    """
    This class performs convolutions of a subset of pixels at higher subsampled resolution
    """


"""

def py_convolve(im, kernel, points):
    ks = kernel.shape[0] // 2
    data = np.pad(im, ks, mode='constant', constant_values=0)
    return cy_convolve(data, kernel, points)



import numpy as np
cimport cython

@cython.boundscheck(False)
def cy_convolve(unsigned char[:, ::1] im, double[:, ::1] kernel, Py_ssize_t[:, ::1] points):
    cdef Py_ssize_t i, j, y, x, n, ks = kernel.shape[0]
    cdef Py_ssize_t npoints = points.shape[0]
    cdef double[::1] responses = np.zeros(npoints, dtype='f8')

    for n in range(npoints):
        y = points[n, 0]
        x = points[n, 1]
        for i in range(ks):
            for j in range(ks):
                responses[n] += im[y+i, x+j] * kernel[i, j]

     return np.asarray(responses)
"""