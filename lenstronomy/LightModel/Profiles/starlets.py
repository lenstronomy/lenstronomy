__author__ = 'aymgal'

# TODO : merge in a clever array reshape operations (pysap2coeffs, cube2array, etc...)

import numpy as np

from lenstronomy.Util import util
from lenstronomy.LightModel.Profiles import starlets_slit


_force_no_pysap = False


class Starlets(object):
    """
    Implementation of the Isotropic Undecimated Walevet Transform (aka "starlet") 
    using the 'a trous' algorithm.

    Based on Starck et al. : https://ui.adsabs.harvard.edu/abs/2007ITIP...16..297S/abstract
    """
    param_names = ['coeffs', 'n_scales', 'n_pixels']
    lower_limit_default = {'coeffs': [0], 'n_scales': 2, 'n_pixels': 10}
    upper_limit_default = {'coeffs': [1e8], 'n_scales': 20, 'n_pixels': 1e10}

    def __init__(self, thread_count=1, fast_inverse=True, second_gen=False, show_pysap_plots=False):
        self.use_pysap, pysap = self._load_pysap()
        if self.use_pysap:
            self._transf_class = pysap.load_transform('BsplineWaveletTransformATrousAlgorithm')
        self._thread_count = thread_count
        self._fast_inverse = fast_inverse
        self._second_gen = second_gen
        self._show_pysap_plots = show_pysap_plots

    def function(self, coeffs, n_scales, n_pixels):
        """return inverse starlet transform from starlet coefficients stored in coeffs"""
        if len(coeffs.shape) == 1:
            coeffs = util.array2cube(coeffs, n_scales, n_pixels)
        return util.image2array(self.function_2d(coeffs, n_scales, n_pixels))

    def function_2d(self, coeffs, n_scales, n_pixels):
        """return inverse starlet transform from starlet coefficients stored in coeffs"""
        if self.use_pysap:
            return self._inverse_transform(coeffs, n_scales)
        else:
            return starlets_slit.inverse_transform(coeffs, fast=self._fast_inverse, 
                                                   second_gen=self._second_gen)

    def decomposition(self, image, n_scales):
        """
        decomposes an image into starlet coefficients, as a 1d array
        :return:
        """
        return util.cube2array(self.decomposition_2d(image, n_scales))

    def decomposition_2d(self, image, n_scales):
        """
        decomposes an image into starlet coefficients
        :return:
        """
        if self.use_pysap:
            coeffs = self._transform(image, n_scales)
        else:
            coeffs = starlets_slit.transform(image, n_scales, second_gen=self._second_gen)
        return coeffs

    def spectral_norm(self, num_pix, n_scales):
        if not hasattr(self, '_spectral_norm') or n_scales != self._n_scales_cache:
            self._spectral_norm = self._compute_spectral_norm(num_pix, n_scales, num_iter=20, tol=1e-10)
            self._n_scales_cache = n_scales
        return self._spectral_norm

    def _inverse_transform(self, coeffs, n_scales):
        """performs inverse starlet transform"""
        self._check_transform_pysap(n_scales)
        if self._fast_inverse and not self._second_gen:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            image = np.sum(coeffs, axis=0)
        else:
            coeffs = self._coeffs2pysap(coeffs)
            self._transf.analysis_data = coeffs
            result = self._transf.synthesis()
            if self._show_pysap_plots:
                result.show()
            image = result.data
        return image

    def _transform(self, image, n_scales):
        """
        decomposes an image into starlets coefficients
        """
        self._check_transform_pysap(n_scales)
        self._transf.data = image
        self._transf.analysis()
        if self._show_pysap_plots:
            self._transf.show()
        coeffs = self._transf.analysis_data
        coeffs = self._pysap2coeffs(coeffs)
        return coeffs

    def _check_transform_pysap(self, n_scales):
        if not hasattr(self, '_transf') or n_scales != self._n_scales:
            self._transf = self._transf_class(nb_scale=n_scales, verbose=False, 
                                              nb_procs=self._thread_count)
            self._n_scales = n_scales

    def _pysap2coeffs(self, coeffs):
        return np.asarray(coeffs)

    def _coeffs2pysap(self, coeffs):
        coeffs_list = []
        for i in range(coeffs.shape[0]):
            coeffs_list.append(coeffs[i, :, :])
        return coeffs_list

    def _compute_spectral_norm(self, num_pix, n_scales, num_iter=20, tol=1e-10):
        """compute spectral norm of the starlet operator"""
        operator = lambda x: self.decomposition(x, n_scales)
        inverse_operator = lambda c: self.function(c, n_scales)
        return util.spectral_norm(num_pix, operator, inverse_operator, num_iter=num_iter, tol=tol)


    def _load_pysap(self):
        if _force_no_pysap:
            return False, None
        try:
            import pysap
        except ImportError:
            return False, None
        else:
            return True, pysap
